#include "Baking\LightmappingManager.h"

#include "Blueberry\Core\Time.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Scene\Components\Light.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Tools\FileHelper.h"

#include "Log.h"
#include "CudaBuffer.h"
#include "CudaBVH.h"
#include "Lightmapping\LightmappingParams.h"
#include "Denoising\DenoisingParams.h"
#include "VecMath.h"
#include "MathHelper.h"
#include "Random.h"

#include <iomanip>
#include <iostream>
#include <thread>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

namespace Blueberry
{
	LightmappingState LightmappingManager::s_LightmappingState = {};

	// Maybe make a list of valid per chart texels instead of texture tiles?
	struct ChartData
	{
		List<bool> mask;
		Vector4 meshBounds;
		Vector4 maskBounds;
		Vector2Int position;
		Vector2Int size;
		uint32_t index;
	};

	struct MeshData
	{
		void Release()
		{
			vertexBuffer.free();
			normalBuffer.free();
			tangentBuffer.free();
			indexBuffer.free();
			CUDA_CHECK(cudaFree((void*)gasOutput));
		}

		Mesh* mesh;
		uint32_t chartCount;
		List<Transform*> transforms;

		CUDABuffer vertexBuffer;
		CUDABuffer normalBuffer;
		CUDABuffer tangentBuffer;
		CUDABuffer indexBuffer;

		OptixTraversableHandle gasHandle;
		CUdeviceptr gasOutput;
	};

	template <typename T>
	struct Record
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		T data;
	};

	#define CLOSEST_HIT_COUNT 3

	struct PassState
	{
		OptixPipelineCompileOptions pipelineCompileOptions = {};
		OptixPipeline pipeline = nullptr;
		OptixModule ptxModule = {};

		OptixProgramGroup raygenProgGroup = 0;
		OptixProgramGroup missOcclusionProgGroup = 0;
		OptixProgramGroup missRadianceProgGroup = 0;
		OptixProgramGroup missShadowProgGroup = 0;
		OptixProgramGroup hitgroupOcclusionProgGroup = 0;
		OptixProgramGroup hitgroupRadianceProgGroup = 0;
		OptixProgramGroup hitgroupShadowProgGroup = 0;

		OptixShaderBindingTable sbt = {};
	};

	struct LightmapperState
	{
		OptixDeviceContext context = nullptr;
		CalculationParams params = {};

		List<MeshData> meshDatas = {};
		OptixTraversableHandle iasHandle = 0;

		PassState pass = {};

		CUmodule denoiserPtxModule = {};
		CUfunction denoiserKernelFirstPass = {};
		CUfunction denoiserKernelpass = {};

		uint32_t launchSize;
		List<uint2> validTexels = {};
		List<uint32_t> atlasMask = {};
		CalculationResult result = {};
		CUDABVH bvh = {};
		unsigned int* completeCounter;

		CUstream stream = 0;
		CUDABuffer instanceMatrices = {};
		CUDABuffer validTexelsBuffer = {};
		CUDABuffer colorBuffer = {};
		CUDABuffer normalBuffer = {};
		CUDABuffer positionBuffer = {};
		CUDABuffer chartIndexBuffer = {};
		CUDABuffer denoisedColorBuffer = {};
		LightmappingParams lightmappingParams = {};
		DenoisingParams denoisingParams = {};
	} s_State = {};

	static CUdeviceptr s_ParamsPtr = 0;
	static CUdeviceptr s_DenoisingParamsPtr = 0;
	static int s_Padding = 0;

	typedef Record<RayGenData> RayGenSbtRecord;
	typedef Record<MissData> MissSbtRecord;
	typedef Record<HitGroupData> HitGroupSbtRecord;

	void ContextLogCb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
	{
		std::cout << message << std::endl;
		//std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
		//	<< message << "\n";
	}

	void CreateContext(LightmapperState& state)
	{
		// Initialize context
		{
			// Initialize CUDA
			CUDA_CHECK(cudaFree(0));

			CUcontext cuCtx = 0;  // zero means take the current context
			OPTIX_CHECK(optixInit());
			OptixDeviceContextOptions options = {};
			options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
			options.logCallbackFunction = &ContextLogCb;
			options.logCallbackLevel = 4;
			OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &state.context));
		}
	}
	
	void InitializeAccels(LightmapperState& state, Scene* scene)
	{
		// Initialize meshes
		{
			Dictionary<ObjectId, uint32_t> existingMeshes = {};

			for (auto& component : scene->GetIterator<MeshRenderer>())
			{
				MeshRenderer* meshRenderer = static_cast<MeshRenderer*>(component.second);
				//AABB bounds = meshRenderer->GetBounds();
				//float scale = (bounds.Extents.x + bounds.Extents.y + bounds.Extents.z) / 3;
				Mesh* mesh = meshRenderer->GetMesh();
				auto it = existingMeshes.find(mesh->GetObjectId());
				if (it != existingMeshes.end())
				{
					//state.meshDatas[it->second].scales.emplace_back(scale);
					state.meshDatas[it->second].transforms.emplace_back(meshRenderer->GetTransform());
				}
				else
				{
					MeshData data = {};
					data.mesh = mesh;
					//data.scales.emplace_back(scale);
					data.transforms.emplace_back(meshRenderer->GetTransform());
					data.vertexBuffer.alloc_and_upload(mesh->GetVertices(), mesh->GetVertexCount());
					data.normalBuffer.alloc_and_upload(mesh->GetNormals(), mesh->GetVertexCount());
					data.tangentBuffer.alloc_and_upload(mesh->GetTangents(), mesh->GetVertexCount());
					data.indexBuffer.alloc_and_upload(mesh->GetIndices(), mesh->GetIndexCount());
					state.meshDatas.emplace_back(data);
				}
			}
		}

		// Build acceleration structures
		{
			OptixAccelBuildOptions accelOptions = {};
			accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
			accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

			for (auto& data : state.meshDatas)
			{
				Mesh* mesh = data.mesh;
				OptixBuildInput buildInput = {};
				unsigned int flags = 1;

				buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
				buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
				buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
				buildInput.triangleArray.numVertices = static_cast<unsigned int>(mesh->GetVertexCount());
				buildInput.triangleArray.vertexBuffers = &data.vertexBuffer.data;
				buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
				buildInput.triangleArray.indexStrideInBytes = sizeof(int3);
				buildInput.triangleArray.numIndexTriplets = static_cast<unsigned int>(mesh->GetIndexCount()) / 3;
				buildInput.triangleArray.indexBuffer = data.indexBuffer.data;
				buildInput.triangleArray.flags = &flags;
				buildInput.triangleArray.numSbtRecords = 1;

				OptixAccelBufferSizes gasBufferSizes;
				OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &accelOptions, &buildInput, 1, &gasBufferSizes));

				CUDA_CHECK(cudaMalloc((void**)&data.gasOutput, gasBufferSizes.outputSizeInBytes));

				CUdeviceptr tempBuffer;
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tempBuffer), gasBufferSizes.tempSizeInBytes));

				OPTIX_CHECK(optixAccelBuild(state.context, 0,   // CUDA stream
					&accelOptions,
					&buildInput,
					1,
					tempBuffer,
					gasBufferSizes.tempSizeInBytes,
					data.gasOutput,
					gasBufferSizes.outputSizeInBytes,
					&data.gasHandle,
					nullptr,  // emitted property list
					0         // num emitted properties
				));

				CUDA_CHECK(cudaFree(reinterpret_cast<void*>(tempBuffer)));
			}
		}

		// Build instance acceleration structure
		CUdeviceptr iasOutput = 0;
		{
			unsigned int instanceCount = 0;
			for (auto& data : state.meshDatas)
			{
				instanceCount += data.transforms.size();
			}
			size_t instanceSizeInBytes = sizeof(OptixInstance) * instanceCount;

			CUdeviceptr instanceBuffer;
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&instanceBuffer), instanceSizeInBytes));

			OptixBuildInput instanceInput = {};

			instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
			instanceInput.instanceArray.instances = instanceBuffer;
			instanceInput.instanceArray.numInstances = instanceCount;

			OptixAccelBuildOptions accelOptions = {};
			accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
			accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

			OptixAccelBufferSizes iasBufferSizes;
			OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &accelOptions, &instanceInput,
				1,  // num build inputs
				&iasBufferSizes));

			CUdeviceptr tempBuffer;
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tempBuffer), iasBufferSizes.tempSizeInBytes));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&iasOutput), iasBufferSizes.outputSizeInBytes));

			OptixInstance* optixInstanceArray = BB_MALLOC_ARRAY(OptixInstance, instanceCount);
			Matrix3x4* instanceMatrixArray = BB_MALLOC_ARRAY(Matrix3x4, instanceCount);
			size_t instanceOffset = 0;

			for (auto& data : state.meshDatas)
			{
				for (Transform* transform : data.transforms)
				{
					Matrix transformMatrix = transform->GetLocalToWorldMatrix();
					float tr[] = 
					{
						transformMatrix._11, transformMatrix._21, transformMatrix._31, transformMatrix._41,
						transformMatrix._12, transformMatrix._22, transformMatrix._32, transformMatrix._42,
						transformMatrix._13, transformMatrix._23, transformMatrix._33, transformMatrix._43,
					};
					
					OptixInstance optixInstance;
					memcpy(optixInstance.transform, tr, sizeof(float) * 12);
					optixInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
					optixInstance.instanceId = static_cast<unsigned int>(instanceOffset);
					optixInstance.sbtOffset = static_cast<unsigned int>(instanceOffset * CLOSEST_HIT_COUNT);
					optixInstance.visibilityMask = 1u;
					optixInstance.traversableHandle = data.gasHandle;

					optixInstanceArray[instanceOffset] = optixInstance;
					instanceMatrixArray[instanceOffset] = Matrix3x4(tr);
					++instanceOffset;
				}
			}

			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(instanceBuffer), optixInstanceArray, instanceSizeInBytes,
				cudaMemcpyHostToDevice));
			state.instanceMatrices.alloc_and_upload(instanceMatrixArray, instanceCount);
			BB_FREE(optixInstanceArray);
			BB_FREE(instanceMatrixArray);

			OPTIX_CHECK(optixAccelBuild(state.context,
				0,  // CUDA stream
				&accelOptions,
				&instanceInput,
				1,  // num build inputs
				tempBuffer,
				iasBufferSizes.tempSizeInBytes,
				iasOutput,
				iasBufferSizes.outputSizeInBytes,
				&state.iasHandle,
				nullptr,  // emitted property list
				0u         // num emitted properties
			));

			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(tempBuffer)));
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(instanceBuffer)));
		}
	}

	void CreatePTXModule(LightmapperState& state)
	{
		// Create PTX module
		{
			OptixModuleCompileOptions moduleCompileOptions = {};
			moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
			moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

			// Pass
			{
				state.pass.pipelineCompileOptions.usesMotionBlur = false;
				state.pass.pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
				state.pass.pipelineCompileOptions.numPayloadValues = 6;
				state.pass.pipelineCompileOptions.numAttributeValues = 2;
				state.pass.pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE; // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
				state.pass.pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

				String ptx;
				FileHelper::Load(ptx, "assets\\ptx\\Lightmapping.ptx");

				char log[2048];
				size_t sizeofLog = sizeof(log);
				OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
					state.context,
					&moduleCompileOptions,
					&state.pass.pipelineCompileOptions,
					ptx.c_str(),
					ptx.size(),
					log,
					&sizeofLog,
					&state.pass.ptxModule
				));
			}
		}
	}

	void CreateDenoiserPTXModule(LightmapperState& state)
	{
		// Create PTX module
		{
			CUDA_CHECK_RESULT(cuModuleLoad(&state.denoiserPtxModule, "assets\\ptx\\Denoising.ptx"));
			CUDA_CHECK_RESULT(cuModuleGetFunction(&state.denoiserKernelFirstPass, state.denoiserPtxModule, "__denoise__firstpass"));
			CUDA_CHECK_RESULT(cuModuleGetFunction(&state.denoiserKernelpass, state.denoiserPtxModule, "__denoise__pass"));
		}
	}

	void CreateProgramGroups(LightmapperState& state)
	{
		// Create program groups
		{
			// Pass
			{
				OptixProgramGroupOptions programGroupOptions = {};

				OptixProgramGroupDesc raygenProgGroupDesc = {};
				raygenProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
				raygenProgGroupDesc.raygen.module = state.pass.ptxModule;
				raygenProgGroupDesc.raygen.entryFunctionName = "__raygen__pass";

				char log[2048];
				size_t sizeofLog = sizeof(log);
				OPTIX_CHECK_LOG(optixProgramGroupCreate(
					state.context,
					&raygenProgGroupDesc,
					1,                             // num program groups
					&programGroupOptions,
					log,
					&sizeofLog,
					&state.pass.raygenProgGroup
				)
				);

				OptixProgramGroupDesc missProgGroupDesc = {};
				missProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
				missProgGroupDesc.miss.module = state.pass.ptxModule;
				missProgGroupDesc.miss.entryFunctionName = "__miss__occlusion";
				sizeofLog = sizeof(log);
				OPTIX_CHECK_LOG(optixProgramGroupCreate(
					state.context,
					&missProgGroupDesc,
					1,                             // num program groups
					&programGroupOptions,
					log,
					&sizeofLog,
					&state.pass.missOcclusionProgGroup
				)
				);

				memset(&missProgGroupDesc, 0, sizeof(OptixProgramGroupDesc));
				missProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
				missProgGroupDesc.miss.module = state.pass.ptxModule;
				missProgGroupDesc.miss.entryFunctionName = "__miss__radiance";
				sizeofLog = sizeof(log);
				OPTIX_CHECK_LOG(optixProgramGroupCreate(
					state.context,
					&missProgGroupDesc,
					1,                             // num program groups
					&programGroupOptions,
					log,
					&sizeofLog,
					&state.pass.missRadianceProgGroup
				)
				);

				memset(&missProgGroupDesc, 0, sizeof(OptixProgramGroupDesc));
				missProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
				missProgGroupDesc.miss.module = state.pass.ptxModule;
				missProgGroupDesc.miss.entryFunctionName = "__miss__shadow";
				sizeofLog = sizeof(log);
				OPTIX_CHECK_LOG(optixProgramGroupCreate(
					state.context,
					&missProgGroupDesc,
					1,                             // num program groups
					&programGroupOptions,
					log,
					&sizeofLog,
					&state.pass.missShadowProgGroup
				)
				);

				OptixProgramGroupDesc hitgroupProgGroupDesc = {};
				hitgroupProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
				hitgroupProgGroupDesc.hitgroup.moduleCH = state.pass.ptxModule;
				hitgroupProgGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
				sizeofLog = sizeof(log);
				OPTIX_CHECK_LOG(optixProgramGroupCreate(
					state.context,
					&hitgroupProgGroupDesc,
					1,                             // num program groups
					&programGroupOptions,
					log,
					&sizeofLog,
					&state.pass.hitgroupOcclusionProgGroup
				)
				);

				memset(&hitgroupProgGroupDesc, 0, sizeof(OptixProgramGroupDesc));
				hitgroupProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
				hitgroupProgGroupDesc.hitgroup.moduleCH = state.pass.ptxModule;
				hitgroupProgGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
				sizeofLog = sizeof(log);
				OPTIX_CHECK_LOG(optixProgramGroupCreate(
					state.context,
					&hitgroupProgGroupDesc,
					1,                             // num program groups
					&programGroupOptions,
					log,
					&sizeofLog,
					&state.pass.hitgroupRadianceProgGroup
				)
				);

				memset(&hitgroupProgGroupDesc, 0, sizeof(OptixProgramGroupDesc));
				hitgroupProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
				hitgroupProgGroupDesc.hitgroup.moduleCH = state.pass.ptxModule;
				hitgroupProgGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
				sizeofLog = sizeof(log);
				OPTIX_CHECK_LOG(optixProgramGroupCreate(
					state.context,
					&hitgroupProgGroupDesc,
					1,                             // num program groups
					&programGroupOptions,
					log,
					&sizeofLog,
					&state.pass.hitgroupShadowProgGroup
				)
				);
			}
		}
	}

	void CreatePipeline(LightmapperState& state)
	{
		// Link Pipeline
		{
			// Pass
			{
				OptixProgramGroup programGroups[] =
				{
					state.pass.raygenProgGroup,
					state.pass.missOcclusionProgGroup,
					state.pass.missRadianceProgGroup,
					state.pass.missShadowProgGroup,
					state.pass.hitgroupOcclusionProgGroup,
					state.pass.hitgroupRadianceProgGroup,
					state.pass.hitgroupShadowProgGroup,
				};

				OptixPipelineLinkOptions pipelineLinkOptions = {};
				pipelineLinkOptions.maxTraceDepth = 8;
				pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

				char log[2048];
				size_t sizeofLog = sizeof(log);
				OPTIX_CHECK_LOG(optixPipelineCreate(
					state.context,
					&state.pass.pipelineCompileOptions,
					&pipelineLinkOptions,
					programGroups,
					sizeof(programGroups) / sizeof(programGroups[0]),
					log,
					&sizeofLog,
					&state.pass.pipeline
				));
			}
		}
	}

	void CreateSBT(LightmapperState& state)
	{
		// Set up shader binding table
		{
			// Pass
			{
				CUdeviceptr raygenRecord = 0;
				const size_t raygenRecordSize = sizeof(RayGenSbtRecord);
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygenRecord), raygenRecordSize));

				RayGenSbtRecord rRecord;
				OPTIX_CHECK(optixSbtRecordPackHeader(state.pass.raygenProgGroup, &rRecord));
				CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygenRecord), &rRecord, raygenRecordSize, cudaMemcpyHostToDevice));

				List<MissSbtRecord> missRecordsList = {};
				{
					MissSbtRecord occlusionRecord = {};
					OPTIX_CHECK(optixSbtRecordPackHeader(state.pass.missOcclusionProgGroup, &occlusionRecord));
					missRecordsList.emplace_back(occlusionRecord);

					MissSbtRecord radianceRecord = {};
					OPTIX_CHECK(optixSbtRecordPackHeader(state.pass.missRadianceProgGroup, &radianceRecord));
					missRecordsList.emplace_back(radianceRecord);

					MissSbtRecord shadowRecord = {};
					OPTIX_CHECK(optixSbtRecordPackHeader(state.pass.missShadowProgGroup, &shadowRecord));
					missRecordsList.emplace_back(shadowRecord);
				}

				List<HitGroupSbtRecord> hitgroupRecordsList = {};
				for (auto& data : state.meshDatas)
				{
					HitGroupSbtRecord occlusionRecord = {};
					occlusionRecord.data.vertices = (float3*)data.vertexBuffer.data;
					occlusionRecord.data.normals = (float3*)data.normalBuffer.data;
					occlusionRecord.data.tangents = (float4*)data.tangentBuffer.data;
					occlusionRecord.data.indices = (uint3*)data.indexBuffer.data;
					OPTIX_CHECK(optixSbtRecordPackHeader(state.pass.hitgroupOcclusionProgGroup, &occlusionRecord));
					hitgroupRecordsList.push_back(occlusionRecord);

					HitGroupSbtRecord radianceRecord = {};
					radianceRecord.data.vertices = (float3*)data.vertexBuffer.data;
					radianceRecord.data.normals = (float3*)data.normalBuffer.data;
					radianceRecord.data.tangents = (float4*)data.tangentBuffer.data;
					radianceRecord.data.indices = (uint3*)data.indexBuffer.data;
					OPTIX_CHECK(optixSbtRecordPackHeader(state.pass.hitgroupRadianceProgGroup, &radianceRecord));
					hitgroupRecordsList.push_back(radianceRecord);

					HitGroupSbtRecord shadowRecord = {};
					shadowRecord.data.vertices = (float3*)data.vertexBuffer.data;
					shadowRecord.data.normals = (float3*)data.normalBuffer.data;
					shadowRecord.data.tangents = (float4*)data.tangentBuffer.data;
					shadowRecord.data.indices = (uint3*)data.indexBuffer.data;
					OPTIX_CHECK(optixSbtRecordPackHeader(state.pass.hitgroupShadowProgGroup, &shadowRecord));
					hitgroupRecordsList.push_back(shadowRecord);
				}

				CUdeviceptr missRecords = 0;
				const size_t missRecordSize = sizeof(MissSbtRecord);
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&missRecords), missRecordSize * missRecordsList.size()));
				CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(missRecords), missRecordsList.data(), missRecordSize * missRecordsList.size(), cudaMemcpyHostToDevice));

				CUdeviceptr hitgroupRecords = 0;
				const size_t hitgroupRecordSize = sizeof(HitGroupSbtRecord);
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroupRecords), hitgroupRecordSize * hitgroupRecordsList.size()));
				CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroupRecords), hitgroupRecordsList.data(), hitgroupRecordSize * hitgroupRecordsList.size(), cudaMemcpyHostToDevice));

				state.pass.sbt.raygenRecord = raygenRecord;
				state.pass.sbt.missRecordBase = missRecords;
				state.pass.sbt.missRecordCount = missRecordsList.size();
				state.pass.sbt.missRecordStrideInBytes = static_cast<uint32_t>(missRecordSize);
				state.pass.sbt.hitgroupRecordBase = hitgroupRecords;
				state.pass.sbt.hitgroupRecordCount = hitgroupRecordsList.size();
				state.pass.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroupRecordSize);
			}
		}
	}

	void InitializeParams(LightmapperState& state)
	{
		CUDA_CHECK(cudaStreamCreate(&state.stream));
		state.lightmappingParams.handle = state.iasHandle;
		state.lightmappingParams.instanceMatrices = reinterpret_cast<Matrix3x4*>(state.instanceMatrices.data);
	}

	bool CompareCharts(ChartData& c1, ChartData& c2)
	{
		return c1.size.x + c1.size.y > c2.size.x + c2.size.y;
	}
	
	void InitializeChartsAndBVH(LightmapperState& state)
	{
		List<ChartData> atlasCharts = {};

		for (auto& data : state.meshDatas)
		{
			Mesh* mesh = data.mesh;
			uint32_t vertexCount = data.mesh->GetVertexCount();
			Vector3* vertices = data.mesh->GetVertices();
			uint32_t indexCount = data.mesh->GetIndexCount();
			uint32_t* indices = data.mesh->GetIndices();
			Vector3* uvs = reinterpret_cast<Vector3*>(mesh->GetUVs(1));

			float maxChart = 0;
			for (uint32_t i = 0; i < mesh->GetVertexCount(); ++i)
			{
				maxChart = std::max(uvs[i].z, maxChart);
			}
			data.chartCount = static_cast<uint32_t>(maxChart + 1);
			List<Vector4> charts = {};
			for (uint32_t i = 0; i < data.chartCount; ++i)
			{
				float chartIndex = static_cast<float>(i);
				Vector4 chartBounds = { FLT_MAX, FLT_MAX, FLT_MIN, FLT_MIN };
				for (uint32_t j = 0; j < mesh->GetVertexCount(); ++j)
				{
					Vector3 uv = uvs[j];
					if (uv.z == chartIndex)
					{
						chartBounds.x = std::min(uv.x, chartBounds.x);
						chartBounds.y = std::min(uv.y, chartBounds.y);
						chartBounds.z = std::max(uv.x, chartBounds.z);
						chartBounds.w = std::max(uv.y, chartBounds.w);
					}
				}
				charts.emplace_back(chartBounds);
			}
			
			float texelPerUnit = state.params.texelPerUnit;
			for (auto& transform : data.transforms)
			{
				if (!transform->GetEntity()->GetComponent<MeshRenderer>()->IsBakeable())
				{
					continue;
				}

				state.result.chartInstanceOffset.insert_or_assign(transform->GetEntity()->GetComponent<MeshRenderer>()->GetObjectId(), atlasCharts.size() + 1);
				Matrix localToWorld = transform->GetLocalToWorldMatrix();
				
				for (uint32_t i = 0; i < data.chartCount; ++i)
				{
					ChartData chartData = {};
					float chartIndex = static_cast<float>(i);
					float chartScale = texelPerUnit;
					for (uint32_t j = 0; j < indexCount; j += 3)
					{
						if (uvs[indices[j]].z == chartIndex)
						{
							Vector3 v1 = Vector3::Transform(vertices[indices[j]], localToWorld);
							Vector3 v2 = Vector3::Transform(vertices[indices[j + 1]], localToWorld);
							Vector3 v3 = Vector3::Transform(vertices[indices[j + 2]], localToWorld);

							Vector3 p1 = uvs[indices[j]];
							Vector3 p2 = uvs[indices[j + 1]];
							Vector3 p3 = uvs[indices[j + 2]];

							float s1 = Vector3::Distance(v1, v2) * texelPerUnit / Vector3::Distance(p1, p2);
							float s2 = Vector3::Distance(v2, v3) * texelPerUnit / Vector3::Distance(p2, p3);
							float s3 = Vector3::Distance(v3, v1) * texelPerUnit / Vector3::Distance(p3, p1);

							chartScale = (s1 + s2 + s3) / 3;
							if (std::isinf(chartScale))
							{
								continue;
							}
							break;
						}
					}

					Vector4 meshChartBounds = charts[i];
					Vector2 chartCenter = (Vector2(meshChartBounds.x, meshChartBounds.y) + Vector2(meshChartBounds.z, meshChartBounds.w)) / 2;
					List<Vector2> atlasVertices = {};
					Vector4 atlasChartBounds = { FLT_MAX, FLT_MAX, FLT_MIN, FLT_MIN };

					for (uint32_t j = 0; j < indexCount; ++j)
					{
						Vector3 uv = uvs[indices[j]];
						if (uv.z == chartIndex)
						{
							Vector2 vertex = Vector2(uv.x - chartCenter.x, uv.y - chartCenter.y) * chartScale;
							atlasChartBounds.x = std::min(vertex.x, atlasChartBounds.x);
							atlasChartBounds.y = std::min(vertex.y, atlasChartBounds.y);
							atlasChartBounds.z = std::max(vertex.x, atlasChartBounds.z);
							atlasChartBounds.w = std::max(vertex.y, atlasChartBounds.w);
							atlasVertices.emplace_back(vertex);
						}
					}

					const float epsilon = 0.5 + s_Padding;
					Vector4Int roundedAtlasChartBounds = Vector4Int(static_cast<int>(std::floorf(atlasChartBounds.x - epsilon)), static_cast<int>(std::floorf(atlasChartBounds.y - epsilon)), static_cast<int>(std::ceilf(atlasChartBounds.z + epsilon)), static_cast<int>(std::ceilf(atlasChartBounds.w + epsilon)));
					chartData.meshBounds = meshChartBounds;
					chartData.maskBounds = atlasChartBounds;
					chartData.size = Vector2Int(roundedAtlasChartBounds.z - roundedAtlasChartBounds.x, roundedAtlasChartBounds.w - roundedAtlasChartBounds.y);
					chartData.mask.resize(chartData.size.x * chartData.size.y);
					chartData.index = atlasCharts.size() + 1;
					Vector2Int halfSize = Vector2Int(chartData.size.x / 2, chartData.size.y / 2);
					uint32_t chartVertexCount = static_cast<uint32_t>(atlasVertices.size());

					for (int x = 0; x < chartData.size.x; ++x)
					{
						for (int y = 0; y < chartData.size.y; ++y)
						{
							Vector4 texelBounds = Vector4(x - halfSize.x - 0.5f - s_Padding, y - halfSize.y - 0.5f - s_Padding, x - halfSize.x + 1.5f + s_Padding, y - halfSize.y + 1.5f + s_Padding);
							bool intersects = false;
							for (uint32_t j = 0; j < chartVertexCount; j += 3)
							{
								if (MathHelper::Intersects(texelBounds, atlasVertices[j], atlasVertices[j + 1], atlasVertices[j + 2]))
								{
									intersects = true;
									break;
								}
							}
							chartData.mask[x + y * chartData.size.x] = intersects;
						}
					}
					atlasCharts.emplace_back(std::move(chartData));
				}
			}
		}
		
		uint32_t offset = 0;
		uint32_t chartOffset = 0;
		List<Vector3> uvs = {};

		int size = state.params.maxSize;
		std::sort(atlasCharts.begin(), atlasCharts.end(), CompareCharts);
		state.atlasMask.resize(size * size);

		uint32_t maskChartOffset = 1;
		uint32_t row = 0;
		uint32_t lastChartSize = 0;

		state.result.chartOffsetScale.resize(atlasCharts.size() + 1);
		for (auto& chart : atlasCharts)
		{
			// Reset to top if chart is smaller
			if (lastChartSize > chart.size.x + chart.size.y)
			{
				row = 0;
			}
			bool placeFound = false;
			uint32_t rows = size - chart.size.y;
			for (uint32_t j = row; j < rows; ++j)
			{
				uint32_t columns = size - chart.size.x;
				for (uint32_t i = 0; i < columns; ++i)
				{
					if (state.atlasMask[j * size + i] == 0)
					{
						// Check chart
						bool canFit = true;
						for (uint32_t y = 0; y < chart.size.y; ++y)
						{
							for (uint32_t x = 0; x < chart.size.x; ++x)
							{
								if (chart.mask[y * chart.size.x + x] && state.atlasMask[(j + y) * size + (i + x)] > 0)
								{
									canFit = false;
									break;
								}
							}
							if (!canFit)
							{
								break;
							}
						}
						if (canFit)
						{
							row = j;
							placeFound = true;
							chart.position = Vector2Int(i, j);
							Vector4 meshChartBounds = chart.meshBounds;
							Vector4 maskChartBounds = chart.maskBounds;

							maskChartBounds.x += (chart.position.x + chart.size.x / 2);
							maskChartBounds.y += (chart.position.y + chart.size.y / 2);
							maskChartBounds.z += (chart.position.x + chart.size.x / 2);
							maskChartBounds.w += (chart.position.y + chart.size.y / 2);

							maskChartBounds.x /= size;
							maskChartBounds.y /= size;
							maskChartBounds.z /= size;
							maskChartBounds.w /= size;

							Vector2 scale = Vector2((maskChartBounds.z - maskChartBounds.x) / (meshChartBounds.z - meshChartBounds.x), (maskChartBounds.w - maskChartBounds.y) / (meshChartBounds.w - meshChartBounds.y));
							Vector2 offset = Vector2(maskChartBounds.x - meshChartBounds.x * scale.x, maskChartBounds.y - meshChartBounds.y * scale.y);
							state.result.chartOffsetScale[chart.index] = Vector4(
								offset.x,
								offset.y,
								scale.x,
								scale.y
							);
							for (uint32_t y = 0; y < chart.size.y; ++y)
							{
								for (uint32_t x = 0; x < chart.size.x; ++x)
								{
									uint32_t index = (j + y) * size + (i + x);
									if (state.atlasMask[index] == 0 && chart.mask[y * chart.size.x + x])
									{
										state.atlasMask[index] = maskChartOffset;
										state.validTexels.emplace_back(make_uint2(i + x, j + y));
									}
								}
							}
							break;
						}
					}
				}
				if (placeFound)
				{
					break;
				}
			}
			lastChartSize = chart.size.x + chart.size.y;
			++maskChartOffset;
		}

		maskChartOffset = 1;
		for (auto& data : state.meshDatas)
		{
			Mesh* mesh = data.mesh;
			uint32_t vertexCount = data.mesh->GetVertexCount();
			Vector3* ligthmapUvs = reinterpret_cast<Vector3*>(mesh->GetUVs(1));

			for (Transform* transform : data.transforms)
			{
				if (!transform->GetEntity()->GetComponent<MeshRenderer>()->IsBakeable())
				{
					continue;
				}

				for (uint32_t i = 0; i < vertexCount; ++i)
				{
					Vector3 uv = ligthmapUvs[i];
					Vector4 scaleOffset = state.result.chartOffsetScale[maskChartOffset + static_cast<uint32_t>(uv.z)];
					uv.x = uv.x * scaleOffset.z + scaleOffset.x;
					uv.y = uv.y * scaleOffset.w + scaleOffset.y;
					uvs.emplace_back(uv);
				}

				CUDABVHInput bvhInput = { uvs.data(), data.mesh->GetIndices(), data.mesh->GetVertexCount(), data.mesh->GetIndexCount(), data.vertexBuffer.data, data.normalBuffer.data, data.tangentBuffer.data, data.indexBuffer.data };
				state.bvh.AddInstance(bvhInput);
				uvs.clear();
				maskChartOffset += data.chartCount;
			}
		}
		state.bvh.Build();
		state.lightmappingParams.bvh = state.bvh.bvh;
		state.validTexelsBuffer.resize(state.validTexels.size() * sizeof(uint2));
		state.validTexelsBuffer.upload(state.validTexels.data(), state.validTexels.size());
		state.lightmappingParams.validTexels = reinterpret_cast<uint2*>(state.validTexelsBuffer.data);
		state.lightmappingParams.validTexelsCount = static_cast<uint32_t>(state.validTexels.size());
		state.chartIndexBuffer.upload(state.atlasMask.data(), state.atlasMask.size());
	}

	void InitializeFrameBuffer(LightmapperState& state, const Vector2Int& viewport)
	{
		state.lightmappingParams.imageSize.x = viewport.x;
		state.lightmappingParams.imageSize.y = viewport.y;
		state.lightmappingParams.samplePerTexel = state.params.samplePerTexel;
		state.lightmappingParams.texelPerUnit = state.params.texelPerUnit;
		state.denoisingParams.imageSize.x = viewport.x;
		state.denoisingParams.imageSize.y = viewport.y;

		size_t frameBufferSize = state.lightmappingParams.imageSize.x * state.lightmappingParams.imageSize.y * sizeof(float4);
		size_t normalBufferSize = state.lightmappingParams.imageSize.x * state.lightmappingParams.imageSize.y * sizeof(float3);
		if (state.lightmappingParams.color == nullptr)
		{
			state.colorBuffer.alloc(frameBufferSize);
			state.normalBuffer.alloc(normalBufferSize);
			state.positionBuffer.alloc(frameBufferSize);
			state.denoisedColorBuffer.alloc(frameBufferSize);
			state.chartIndexBuffer.alloc(frameBufferSize / 4);
			state.lightmappingParams.color = reinterpret_cast<float4*>(state.colorBuffer.data);
			state.lightmappingParams.normal = reinterpret_cast<float3*>(state.normalBuffer.data);
			state.lightmappingParams.position = reinterpret_cast<float4*>(state.positionBuffer.data);
			state.denoisingParams.inputColor = reinterpret_cast<float4*>(state.colorBuffer.data);
			state.denoisingParams.inputNormal = reinterpret_cast<float3*>(state.normalBuffer.data);
			state.denoisingParams.inputPosition = reinterpret_cast<float4*>(state.positionBuffer.data);
			state.denoisingParams.outputColor = reinterpret_cast<float4*>(state.denoisedColorBuffer.data);
			state.denoisingParams.chartIndex = reinterpret_cast<unsigned int*>(state.chartIndexBuffer.data);
		}
		else if (state.colorBuffer.sizeInBytes != frameBufferSize)
		{
			state.colorBuffer.resize(frameBufferSize);
			state.chartIndexBuffer.resize(frameBufferSize / 4);
			state.normalBuffer.resize(normalBufferSize);
			state.positionBuffer.resize(frameBufferSize);
			state.denoisedColorBuffer.resize(frameBufferSize);
		}
	}

	void InitializeLights(LightmapperState& state, Scene* scene)
	{
		for (auto component : scene->GetIterator<Light>())
		{
			Light* light = static_cast<Light*>(component.second);
			if (light->GetType() == LightType::Directional)
			{
				Transform* transform = light->GetTransform();
				Vector3 dir = Vector3::Transform(Vector3::Backward, transform->GetRotation());
				dir.Normalize();
				Color color = light->GetColor();
				float intensity = light->GetIntensity();
				state.lightmappingParams.directionalLight.direction = { dir.x, dir.y, dir.z };
				state.lightmappingParams.directionalLight.color = { color.R() * intensity, color.G() * intensity, color.B() * intensity };
			}
		}
	}

	void Launch(LightmapperState& state)
	{
		// Launch
		{
			cudaHostAlloc(&s_State.completeCounter, sizeof(unsigned int), cudaHostAllocMapped | cudaHostAllocPortable);
			cudaHostGetDevicePointer(&s_State.lightmappingParams.completeCounter, s_State.completeCounter, 0);

			if (s_ParamsPtr == 0)
			{
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_ParamsPtr), sizeof(LightmappingParams)));
			}

			CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void*>(s_ParamsPtr),
				&state.lightmappingParams, sizeof(LightmappingParams),
				cudaMemcpyHostToDevice
			));

			OPTIX_CHECK(optixLaunch(state.pass.pipeline, state.stream, s_ParamsPtr, sizeof(LightmappingParams), &state.pass.sbt, state.launchSize, 1, 1));
			cudaError_t err = cudaGetLastError();
			CUDA_SYNC_CHECK();
			
			cudaFreeHost(s_State.completeCounter);
			s_State.completeCounter = nullptr;
			s_State.lightmappingParams.completeCounter = nullptr;
		}
	}

	void LaunchDenoiser(LightmapperState& state)
	{
		// Launch
		{
			if (s_DenoisingParamsPtr == 0)
			{
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_DenoisingParamsPtr), sizeof(DenoisingParams)));
			}

			size_t size;
			CUDA_CHECK_RESULT(cuModuleGetGlobal(&s_DenoisingParamsPtr, &size, state.denoiserPtxModule, "params"));

			CUDA_CHECK_RESULT(cuMemcpyHtoD(s_DenoisingParamsPtr, &state.denoisingParams, sizeof(DenoisingParams)));
			CUDA_CHECK_RESULT(cuLaunchKernel(state.denoiserKernelFirstPass, state.lightmappingParams.imageSize.x / 8, state.lightmappingParams.imageSize.y / 8, 1, 8, 8, 1, 0, 0, 0, 0));
			CUDA_CHECK_RESULT(cuCtxSynchronize());

			float4* input = state.denoisingParams.inputColor;
			state.denoisingParams.inputColor = state.denoisingParams.outputColor;
			state.denoisingParams.outputColor = input;

			CUDA_CHECK_RESULT(cuMemcpyHtoD(s_DenoisingParamsPtr, &state.denoisingParams, sizeof(DenoisingParams)));
			CUDA_CHECK_RESULT(cuLaunchKernel(state.denoiserKernelpass, state.lightmappingParams.imageSize.x / 8, state.lightmappingParams.imageSize.y / 8, 1, 8, 8, 1, 0, 0, 0, 0));
			CUDA_CHECK_RESULT(cuCtxSynchronize());
		}
	}

	void LightmappingManager::Clear()
	{
		if (s_State.context == nullptr)
		{
			return;
		}
		s_State.lightmappingParams.accumulationFrameIndex = 0;
		//s_State.accumulatedFrameBuffer.clear();
	}

	void FixCharts(Vector4* temp, Vector4* result, const Vector2Int& resultSize)
	{
		int atlasSize = static_cast<int>(s_State.atlasMask.size());
		for (int j = 0; j < resultSize.y; ++j)
		{
			for (int i = 0; i < resultSize.x; ++i)
			{
				int index = j * resultSize.x + i;
				Vector4 center = *(temp + index);
				uint32_t centerChartIndex = s_State.atlasMask[index];

				if (center.w > 0.5)
				{
					*(result + index) = center;
				}
				else
				{
					static Vector2Int offsets[8] = { Vector2Int(-1, 0), Vector2Int(0, -1), Vector2Int(1, 0), Vector2Int(0, 1), Vector2Int(-1, -1), Vector2Int(1, -1), Vector2Int(-1, 1), Vector2Int(1, 1) };
					for (uint32_t k = 0; k < 8; ++k)
					{
						Vector2Int offset = offsets[k];
						int nearbyIndex = (j + offset.y) * resultSize.x + (i + offset.x);
						if (nearbyIndex > 0 && nearbyIndex < atlasSize)
						{
							Vector4 nearby = *(temp + nearbyIndex);
							uint32_t nearbyChartIndex = s_State.atlasMask[nearbyIndex];
							if (nearby.w > 0.5 && nearbyChartIndex == centerChartIndex)
							{
								*(result + index) = nearby;
								break;
							}
						}
					}
				}
			}
		}

		for (int j = 0; j < resultSize.y; j += 4)
		{
			for (int i = 0; i < resultSize.x; i += 4)
			{
				bool anyValid = false, anyInvalid = false;
				Vector4 mean = Vector4::Zero;
				int validCount = 0;
				for (int x = 0; x < 4; ++x)
				{
					for (int y = 0; y < 4; ++y)
					{
						int index = (j + y) * resultSize.x + (i + x);
						Vector4 color = temp[index];
						if (color.w > 0.5)
						{
							anyValid = true;
							mean += color;
							++validCount;
						}
						else
						{
							anyInvalid = true;
						}
					}
				}
				if (anyValid && anyInvalid)
				{
					mean /= validCount;
					for (int x = 0; x < 4; ++x)
					{
						for (int y = 0; y < 4; ++y)
						{
							int index = (j + y) * resultSize.x + (i + x);
							if (temp[index].w < 0.5)
							{
								temp[index] = mean;
							}
						}
					}
				}
			}
		}
	}

	void LightmappingManager::Calculate(Scene* scene, const CalculationParams& params)
	{
		if (s_LightmappingState != LightmappingState::None)
		{
			return;
		}

		s_LightmappingState = LightmappingState::Calculating;
		std::thread worker([scene, params]()
		{
			BB_INITIALIZE_ALLOCATOR_THREAD();
			s_State = {};
			s_State.params = params;

			CreateContext(s_State);
			//CreateOptixDenoiser(s_State);
			CreateDenoiserPTXModule(s_State);
			InitializeAccels(s_State, scene);
			CreatePTXModule(s_State);
			CreateProgramGroups(s_State);
			CreatePipeline(s_State);
			CreateSBT(s_State);
			InitializeParams(s_State);

			int size = params.maxSize;
			s_State.result.output.resize(size * size * (sizeof(Vector4) / sizeof(uint8_t)));
			s_State.result.outputSize = Vector2Int(size, size);
			Vector4* temp = BB_MALLOC_ARRAY(Vector4, size * size);

			InitializeFrameBuffer(s_State, s_State.result.outputSize);
			//InitializeOptixDenoiserFrameBuffer(s_State);
			InitializeLights(s_State, scene);
			InitializeChartsAndBVH(s_State);

			uint32_t tileSize = static_cast<uint32_t>(params.tileSize * params.tileSize);
			s_State.launchSize = tileSize;
			BB_INFO("Started baking");
			Launch(s_State);
			BB_INFO("Ended baking");

			if (params.denoise)
			{
				LaunchDenoiser(s_State);
				//LaunchOptixDenoiser(s_State);
				s_State.colorBuffer.download(temp, s_State.colorBuffer.sizeInBytes / sizeof(Vector4));
			}
			else
			{
				s_State.colorBuffer.download(temp, s_State.colorBuffer.sizeInBytes / sizeof(Vector4));
			}

			FixCharts(temp, reinterpret_cast<Vector4*>(s_State.result.output.data()), s_State.result.outputSize);

			BB_FREE(temp);
			BB_SHUTDOWN_ALLOCATOR_THREAD();
			s_LightmappingState = LightmappingState::Waiting;
		});
		worker.detach();
	}

	void LightmappingManager::Shutdown()
	{
		// Release denoiser
		{
			//optixDenoiserDestroy(s_State.denoiser);
		}

		// Release context
		{
			for (auto& data : s_State.meshDatas)
			{
				data.Release();
			}
			s_State.instanceMatrices.free();
			//s_State.accumulatedFrameBuffer.free();
			s_State.colorBuffer.free();
			//s_State.albedoBuffer.free();
			//s_State.normalBuffer.free();
			//s_State.denoisedColorBuffer.free();
			//s_State.denoiserIntensity.free();
			//s_State.denoiserScratch.free();
			//s_State.denoiserState.free();

			OPTIX_CHECK(optixDeviceContextDestroy(s_State.context));
			cuModuleUnload(s_State.denoiserPtxModule);
			s_State = {};
		}
		s_LightmappingState = LightmappingState::None;
	}

	float LightmappingManager::GetProgress()
	{
		if (s_LightmappingState != LightmappingState::Calculating)
		{
			return 0;
		}
		if (s_State.completeCounter == nullptr)
		{
			return 0;
		}
		return static_cast<float>(*s_State.completeCounter) / s_State.validTexels.size();
	}

	const LightmappingState& LightmappingManager::GetLightmappingState()
	{
		return s_LightmappingState;
	}

	CalculationResult& LightmappingManager::GetCalculationResult()
	{
		return s_State.result;
	}
}
