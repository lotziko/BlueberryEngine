#include "Baking\LightmappingManager.h"

#include "Blueberry\Core\Time.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Scene\Components\Light.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Camera.h"
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

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

namespace Blueberry
{
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

	struct LightmapperState
	{
		OptixDeviceContext context = nullptr;

		/*OptixDenoiser denoiser = nullptr;
		OptixDenoiserParams denoiserParams = {};
		OptixImage2D denoiserInputs[3] = {};
		OptixImage2D denoiserOutput = {};
		CUDABuffer denoiserIntensity = {};
		CUDABuffer denoiserScratch = {};
		CUDABuffer denoiserState = {};*/

		List<MeshData> meshDatas = {};
		OptixTraversableHandle iasHandle = 0;

		OptixPipelineCompileOptions pipelineCompileOptions = {};
		OptixPipeline pipeline = nullptr;
		OptixModule ptxModule = {};

		OptixProgramGroup raygenProgGroup = 0;
		OptixProgramGroup missProgGroup = 0;
		OptixProgramGroup hitgroupDefaultProgGroup = 0;
		OptixProgramGroup hitgroupShadowProgGroup = 0;

		CUmodule denoiserPtxModule = {};
		CUfunction denoiserKernel = {};

		uint2 launchSize = {};
		List<uint32_t> atlasMask = {};
		List<Vector4> chartOffsetScale = {};
		Dictionary<ObjectId, uint32_t> chartInstanceOffset = {};
		CUDABVH bvh = {};

		CUstream stream = 0;
		CUDABuffer instanceMatrices = {};
		CUDABuffer accumulatedFrameBuffer = {};
		CUDABuffer colorBuffer = {};
		CUDABuffer albedoBuffer = {};
		CUDABuffer normalBuffer = {};
		CUDABuffer chartIndexBuffer = {};
		CUDABuffer denoisedColorBuffer = {};
		LightmappingParams lightmappingParams = {};
		DenoisingParams denoisingParams = {};
		
		OptixShaderBindingTable sbt = {};
	} s_State = {};

	static CUdeviceptr s_ParamsPtr = 0;
	static CUdeviceptr s_DenoisingParamsPtr = 0;

	static uint32_t s_TexelsPerMeter = 5;
	static uint32_t s_Size = 2048;

	typedef Record<RayGenData> RayGenSbtRecord;
	typedef Record<MissData> MissSbtRecord;
	typedef Record<HitGroupData> HitGroupSbtRecord;

	void ContextLogCb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
	{
		std::cout << message << std::endl;
		//std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
		//	<< message << "\n";
	}

	void GetCameraParams(Camera* camera, float3& camEye, float3& U, float3& V, float3& W)
	{
		Transform* transform = camera->GetTransform();
		Vector3 cameraPosition = transform->GetPosition();
		Vector3 cameraDirection = Vector3::Transform(Vector3::Forward, transform->GetRotation());
		Vector3 cameraUp = Vector3::Transform(Vector3::Up, transform->GetRotation());

		camEye = { cameraPosition.x, cameraPosition.y, cameraPosition.z };
		W = { cameraDirection.x, cameraDirection.y, cameraDirection.z }; // Do not normalize W -- it implies focal length
		float wlen = length(W);
		U = normalize(cross(W, { cameraUp.x, cameraUp.y, cameraUp.z }));
		V = normalize(cross(U, W));

		float vlen = wlen * tanf(0.5f * camera->GetFieldOfView() * M_PIf / 180.0f);
		V = V * vlen;
		float ulen = vlen * camera->GetAspectRatio();
		U = U * ulen;
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

	//void CreateDenoiser(LightmapperState& state)
	//{
	//	// Create denoiser
	//	{
	//		OptixDenoiserOptions options = {};
	//		options.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL;
	//		OPTIX_CHECK(optixDenoiserCreate(state.context, &options, &state.denoiser));
	//		OPTIX_CHECK(optixDenoiserSetModel(
	//			state.denoiser,
	//			OPTIX_DENOISER_MODEL_KIND_HDR,
	//			nullptr, // data
	//			0        // size
	//		));
	//	}
	//}

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
					optixInstance.sbtOffset = static_cast<unsigned int>(instanceOffset * 2);
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

			state.pipelineCompileOptions.usesMotionBlur = false;
			state.pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
			state.pipelineCompileOptions.numPayloadValues = 6;
			state.pipelineCompileOptions.numAttributeValues = 2;
			state.pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE; // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
			state.pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

			String ptx;
			FileHelper::Load(ptx, "assets\\ptx\\Lightmapping.ptx");

			char log[2048];
			size_t sizeofLog = sizeof(log);
			OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
				state.context,
				&moduleCompileOptions,
				&state.pipelineCompileOptions,
				ptx.c_str(),
				ptx.size(),
				log,
				&sizeofLog,
				&state.ptxModule
			));
		}
	}

	void CreateDenoiserPTXModule(LightmapperState& state)
	{
		// Create PTX module
		{
			CUDA_CHECK_RESULT(cuModuleLoad(&state.denoiserPtxModule, "assets\\ptx\\Denoising.ptx"));
			CUDA_CHECK_RESULT(cuModuleGetFunction(&state.denoiserKernel, state.denoiserPtxModule, "__denoise"));
		}
	}

	void CreateProgramGroups(LightmapperState& state)
	{
		// Create program groups
		{
			OptixProgramGroupOptions programGroupOptions = {};

			OptixProgramGroupDesc raygenProgGroupDesc = {};
			raygenProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
			raygenProgGroupDesc.raygen.module = state.ptxModule;
			raygenProgGroupDesc.raygen.entryFunctionName = "__raygen__default";

			char log[2048];
			size_t sizeofLog = sizeof(log);
			OPTIX_CHECK_LOG(optixProgramGroupCreate(
				state.context,
				&raygenProgGroupDesc,
				1,                             // num program groups
				&programGroupOptions,
				log,
				&sizeofLog,
				&state.raygenProgGroup
			)
			);

			OptixProgramGroupDesc missProgGroupDesc = {};
			missProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
			missProgGroupDesc.miss.module = state.ptxModule;
			missProgGroupDesc.miss.entryFunctionName = "__miss__default";
			sizeofLog = sizeof(log);
			OPTIX_CHECK_LOG(optixProgramGroupCreate(
				state.context,
				&missProgGroupDesc,
				1,                             // num program groups
				&programGroupOptions,
				log,
				&sizeofLog,
				&state.missProgGroup
			)
			);

			OptixProgramGroupDesc hitgroupProgGroupDesc = {};
			hitgroupProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			hitgroupProgGroupDesc.hitgroup.moduleCH = state.ptxModule;
			hitgroupProgGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__default";
			sizeofLog = sizeof(log);
			OPTIX_CHECK_LOG(optixProgramGroupCreate(
				state.context,
				&hitgroupProgGroupDesc,
				1,                             // num program groups
				&programGroupOptions,
				log,
				&sizeofLog,
				&state.hitgroupDefaultProgGroup
			)
			);

			memset(&hitgroupProgGroupDesc, 0, sizeof(OptixProgramGroupDesc));
			hitgroupProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			hitgroupProgGroupDesc.hitgroup.moduleCH = state.ptxModule;
			hitgroupProgGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
			sizeofLog = sizeof(log);
			OPTIX_CHECK_LOG(optixProgramGroupCreate(
				state.context,
				&hitgroupProgGroupDesc,
				1,                             // num program groups
				&programGroupOptions,
				log,
				&sizeofLog,
				&state.hitgroupShadowProgGroup
			)
			);
		}
	}

	void CreatePipeline(LightmapperState& state)
	{
		// Link Pipeline
		{
			OptixProgramGroup programGroups[] =
			{
				state.raygenProgGroup,
				state.missProgGroup,
				state.hitgroupDefaultProgGroup,
				state.hitgroupShadowProgGroup
			};

			OptixPipelineLinkOptions pipelineLinkOptions = {};
			pipelineLinkOptions.maxTraceDepth = 8;
			pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

			char log[2048];
			size_t sizeofLog = sizeof(log);
			OPTIX_CHECK_LOG(optixPipelineCreate(
				state.context,
				&state.pipelineCompileOptions,
				&pipelineLinkOptions,
				programGroups,
				sizeof(programGroups) / sizeof(programGroups[0]),
				log,
				&sizeofLog,
				&state.pipeline
			));
		}
	}

	void CreateSBT(LightmapperState& state)
	{
		// Set up shader binding table
		{
			CUdeviceptr raygenRecord = 0;
			const size_t raygenRecordSize = sizeof(RayGenSbtRecord);
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygenRecord), raygenRecordSize));

			RayGenSbtRecord rRecord;
			OPTIX_CHECK(optixSbtRecordPackHeader(state.raygenProgGroup, &rRecord));
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygenRecord), &rRecord, raygenRecordSize, cudaMemcpyHostToDevice));

			CUdeviceptr missRecord = 0;
			const size_t missRecordSize = sizeof(MissSbtRecord);
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&missRecord), missRecordSize));

			MissSbtRecord mRecord;
			OPTIX_CHECK(optixSbtRecordPackHeader(state.missProgGroup, &mRecord));
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(missRecord), &mRecord, missRecordSize, cudaMemcpyHostToDevice));

			List<HitGroupSbtRecord> hitgroupRecordsList = {};
			for (auto& data : state.meshDatas)
			{
				HitGroupSbtRecord defaultRecord = {};
				defaultRecord.data.vertices = (float3*)data.vertexBuffer.data;
				defaultRecord.data.normals = (float3*)data.normalBuffer.data;
				defaultRecord.data.tangents = (float4*)data.tangentBuffer.data;
				defaultRecord.data.indices = (uint3*)data.indexBuffer.data;
				OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroupDefaultProgGroup, &defaultRecord));
				hitgroupRecordsList.push_back(defaultRecord);

				HitGroupSbtRecord shadowRecord = {};
				shadowRecord.data.vertices = (float3*)data.vertexBuffer.data;
				shadowRecord.data.normals = (float3*)data.normalBuffer.data;
				shadowRecord.data.tangents = (float4*)data.tangentBuffer.data;
				shadowRecord.data.indices = (uint3*)data.indexBuffer.data;
				OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroupShadowProgGroup, &shadowRecord));
				hitgroupRecordsList.push_back(shadowRecord);
			}

			CUdeviceptr hitgroupRecords = 0;
			const size_t hitgroupRecordSize = sizeof(HitGroupSbtRecord);
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroupRecords), hitgroupRecordSize * hitgroupRecordsList.size()));
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroupRecords), hitgroupRecordsList.data(), hitgroupRecordSize * hitgroupRecordsList.size(), cudaMemcpyHostToDevice));

			state.sbt.raygenRecord = raygenRecord;
			state.sbt.missRecordBase = missRecord;
			state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(missRecordSize);
			state.sbt.missRecordCount = 1;
			state.sbt.hitgroupRecordBase = hitgroupRecords;
			state.sbt.hitgroupRecordCount = hitgroupRecordsList.size();
			state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroupRecordSize);
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
			
			for (auto& transform : data.transforms)
			{
				state.chartInstanceOffset.insert_or_assign(transform->GetEntity()->GetComponent<MeshRenderer>()->GetObjectId(), atlasCharts.size() + 1);
				Matrix localToWorld = transform->GetLocalToWorldMatrix();
				
				for (uint32_t i = 0; i < data.chartCount; ++i)
				{
					ChartData chartData = {};
					float chartIndex = static_cast<float>(i);
					float chartScale = s_TexelsPerMeter;
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

							float s1 = Vector3::Distance(v1, v2) * s_TexelsPerMeter / Vector3::Distance(p1, p2);
							float s2 = Vector3::Distance(v2, v3) * s_TexelsPerMeter / Vector3::Distance(p2, p3);
							float s3 = Vector3::Distance(v3, v1) * s_TexelsPerMeter / Vector3::Distance(p3, p1);

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

					const float epsilon = 0.5;
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
							Vector4 texelBounds = Vector4(x - halfSize.x - 0.5f, y - halfSize.y - 0.5f, x - halfSize.x + 1.5f, y - halfSize.y + 1.5f);
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

		std::sort(atlasCharts.begin(), atlasCharts.end(), CompareCharts);
		state.atlasMask.resize(s_Size * s_Size);

		uint32_t maskChartOffset = 1;
		uint32_t row = 0;
		uint32_t lastChartSize = 0;

		state.chartOffsetScale.resize(atlasCharts.size() + 1);
		for (auto& chart : atlasCharts)
		{
			// Reset to top if chart is smaller
			if (lastChartSize > chart.size.x + chart.size.y)
			{
				row = 0;
			}
			bool placeFound = false;
			uint32_t rows = s_Size - chart.size.y;
			for (uint32_t j = row; j < rows; ++j)
			{
				uint32_t columns = s_Size - chart.size.x;
				for (uint32_t i = 0; i < columns; ++i)
				{
					if (state.atlasMask[j * s_Size + i] == 0)
					{
						// Check chart
						bool canFit = true;
						for (uint32_t y = 0; y < chart.size.y; ++y)
						{
							for (uint32_t x = 0; x < chart.size.x; ++x)
							{
								if (chart.mask[y * chart.size.x + x] && state.atlasMask[(j + y) * s_Size + (i + x)] > 0)
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

							maskChartBounds.x /= s_Size;
							maskChartBounds.y /= s_Size;
							maskChartBounds.z /= s_Size;
							maskChartBounds.w /= s_Size;

							Vector2 scale = Vector2((maskChartBounds.z - maskChartBounds.x) / (meshChartBounds.z - meshChartBounds.x), (maskChartBounds.w - maskChartBounds.y) / (meshChartBounds.w - meshChartBounds.y));
							Vector2 offset = Vector2(maskChartBounds.x - meshChartBounds.x * scale.x, maskChartBounds.y - meshChartBounds.y * scale.y);
							state.chartOffsetScale[chart.index] = Vector4(
								offset.x,
								offset.y,
								scale.x,
								scale.y
							);
							for (uint32_t y = 0; y < chart.size.y; ++y)
							{
								for (uint32_t x = 0; x < chart.size.x; ++x)
								{
									uint32_t index = (j + y) * s_Size + (i + x);
									if (state.atlasMask[index] == 0 && chart.mask[y * chart.size.x + x])
									{
										state.atlasMask[index] = maskChartOffset;
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
				for (uint32_t i = 0; i < vertexCount; ++i)
				{
					Vector3 uv = ligthmapUvs[i];
					Vector4 scaleOffset = state.chartOffsetScale[maskChartOffset + static_cast<uint32_t>(uv.z)];
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
		state.chartIndexBuffer.upload(state.atlasMask.data(), state.atlasMask.size());

		//Vector4* chartMask = BB_MALLOC_ARRAY(Vector4, s_Size * s_Size);
		//Vector4* normalMask = BB_MALLOC_ARRAY(Vector4, s_Size * s_Size);
		//for (uint32_t i = 0; i < s_Size; ++i)
		//{
		//	for (uint32_t j = 0; j < s_Size; ++j)
		//	{
		//		uint32_t index = j * s_Size + i;
		//		uint32_t chartIndex = state.atlasMask[index];
		//		if (chartIndex > 0)
		//		{
		//			uint32_t seed = chartIndex;
		//			//uint32_t color = /*((float)state.atlasMask[index] / maskChartOffset)*/rnd(chartIndex) * UINT32_MAX;
		//			chartMask[index] = Vector4(rnd(seed), rnd(seed), rnd(seed), 1);//Vector4((color & 0xFF) / 255.0f, ((color >> 8) & 0xFF) / 255.0f, ((color >> 16) & 0xFF) / 255.0f, 1.0f);
		//			normalMask[index] = Vector4(rnd(seed) * 2 - 1, rnd(seed) * 2 - 1, rnd(seed) * 2 - 1, 1);
		//		}
		//	}
		//}
		//state.albedoBuffer.upload(chartMask, s_Size * s_Size);
		//state.normalBuffer.upload(normalMask, s_Size * s_Size);
		//BB_FREE(chartMask);
		//BB_FREE(normalMask);
	}

	void InitializeFrameBufferAndCamera(LightmapperState& state, Camera* camera, const Vector2Int& viewport)
	{
		state.lightmappingParams.accumulationFrameIndex += 1;
		state.lightmappingParams.imageSize.x = viewport.x;
		state.lightmappingParams.imageSize.y = viewport.y;
		GetCameraParams(camera, state.lightmappingParams.camEye, state.lightmappingParams.camU, state.lightmappingParams.camV, state.lightmappingParams.camW);

		size_t accumulatedFrameBufferSize = state.lightmappingParams.imageSize.x * state.lightmappingParams.imageSize.y * sizeof(float4);
		size_t frameBufferSize = state.lightmappingParams.imageSize.x * state.lightmappingParams.imageSize.y * sizeof(float4);
		if (state.lightmappingParams.color == nullptr)
		{
			state.accumulatedFrameBuffer.alloc(accumulatedFrameBufferSize);
			state.lightmappingParams.accumulatedImage = reinterpret_cast<float4*>(state.accumulatedFrameBuffer.data);
			state.colorBuffer.alloc(frameBufferSize);
			state.albedoBuffer.alloc(frameBufferSize);
			state.normalBuffer.alloc(frameBufferSize);
			state.denoisedColorBuffer.alloc(frameBufferSize);
			state.lightmappingParams.color = reinterpret_cast<float4*>(state.colorBuffer.data);
			state.lightmappingParams.normal = reinterpret_cast<float4*>(state.normalBuffer.data);
		}
		else if (state.colorBuffer.sizeInBytes != frameBufferSize)
		{
			state.accumulatedFrameBuffer.resize(accumulatedFrameBufferSize);
			state.colorBuffer.resize(frameBufferSize);
			state.albedoBuffer.resize(frameBufferSize);
			state.normalBuffer.resize(frameBufferSize);
			state.denoisedColorBuffer.resize(frameBufferSize);
		}
	}

	void InitializeFrameBuffer(LightmapperState& state, const Vector2Int& viewport)
	{
		//state.lightmappingParams.accumulationFrameIndex += 1;
		state.lightmappingParams.imageSize.x = viewport.x;
		state.lightmappingParams.imageSize.y = viewport.y;
		state.denoisingParams.imageSize.x = viewport.x;
		state.denoisingParams.imageSize.y = viewport.y;

		//size_t accumulatedFrameBufferSize = state.lightmappingParams.imageSize.x * state.lightmappingParams.imageSize.y * sizeof(float4);
		size_t frameBufferSize = state.lightmappingParams.imageSize.x * state.lightmappingParams.imageSize.y * sizeof(float4);
		if (state.lightmappingParams.color == nullptr)
		{
			//state.accumulatedFrameBuffer.alloc(accumulatedFrameBufferSize);
			//state.lightmappingParams.accumulatedImage = reinterpret_cast<float4*>(state.accumulatedFrameBuffer.data);
			state.colorBuffer.alloc(frameBufferSize);
			//state.albedoBuffer.alloc(frameBufferSize);
			state.normalBuffer.alloc(frameBufferSize);
			state.denoisedColorBuffer.alloc(frameBufferSize);
			state.chartIndexBuffer.alloc(frameBufferSize / 4);
			state.lightmappingParams.color = reinterpret_cast<float4*>(state.colorBuffer.data);
			state.lightmappingParams.normal = reinterpret_cast<float4*>(state.normalBuffer.data);
			state.denoisingParams.inputColor = reinterpret_cast<float4*>(state.colorBuffer.data);
			state.denoisingParams.inputNormal = reinterpret_cast<float4*>(state.normalBuffer.data);
			state.denoisingParams.outputColor = reinterpret_cast<float4*>(state.denoisedColorBuffer.data);
			state.denoisingParams.chartIndex = reinterpret_cast<unsigned int*>(state.chartIndexBuffer.data);
		}
		else if (state.colorBuffer.sizeInBytes != frameBufferSize)
		{
			//state.accumulatedFrameBuffer.resize(accumulatedFrameBufferSize);
			state.colorBuffer.resize(frameBufferSize);
			state.chartIndexBuffer.resize(frameBufferSize / 4);
			//state.albedoBuffer.resize(frameBufferSize);
			state.normalBuffer.resize(frameBufferSize);
			state.denoisedColorBuffer.resize(frameBufferSize);
		}
	}

	//void InitializeDenoiserFrameBuffer(LightmapperState& state)
	//{
	//	// Allocate memory
	//	bool setup = false;
	//	{
	//		OptixDenoiserSizes denoiserSizes;
	//		OPTIX_CHECK(optixDenoiserComputeMemoryResources(
	//			state.denoiser,
	//			state.lightmappingParams.imageSize.x,
	//			state.lightmappingParams.imageSize.y,
	//			&denoiserSizes
	//		));

	//		if (state.denoiserScratch.sizeInBytes == 0)
	//		{
	//			setup = true;
	//			state.denoiserIntensity.alloc(sizeof(float));
	//			state.denoiserScratch.alloc(denoiserSizes.withoutOverlapScratchSizeInBytes);
	//			state.denoiserState.alloc(denoiserSizes.stateSizeInBytes);
	//		}
	//		else if (state.denoiserScratch.sizeInBytes != denoiserSizes.withoutOverlapScratchSizeInBytes)
	//		{
	//			state.denoiserScratch.resize(denoiserSizes.withoutOverlapScratchSizeInBytes);
	//			state.denoiserState.resize(denoiserSizes.stateSizeInBytes);
	//		}
	//	}

	//	// Setup images
	//	{
	//		state.denoiserInputs[0].data = state.colorBuffer.data;
	//		state.denoiserInputs[0].format = OptixPixelFormat::OPTIX_PIXEL_FORMAT_FLOAT4;
	//		state.denoiserInputs[0].width = state.lightmappingParams.imageSize.x;
	//		state.denoiserInputs[0].height = state.lightmappingParams.imageSize.y;
	//		state.denoiserInputs[0].pixelStrideInBytes = sizeof(float4);
	//		state.denoiserInputs[0].rowStrideInBytes = state.lightmappingParams.imageSize.x * sizeof(float4);

	//		state.denoiserInputs[1].data = state.albedoBuffer.data;
	//		state.denoiserInputs[1].format = OptixPixelFormat::OPTIX_PIXEL_FORMAT_FLOAT4;
	//		state.denoiserInputs[1].width = state.lightmappingParams.imageSize.x;
	//		state.denoiserInputs[1].height = state.lightmappingParams.imageSize.y;
	//		state.denoiserInputs[1].pixelStrideInBytes = sizeof(float4);
	//		state.denoiserInputs[1].rowStrideInBytes = state.lightmappingParams.imageSize.x * sizeof(float4);

	//		state.denoiserInputs[2].data = state.normalBuffer.data;
	//		state.denoiserInputs[2].format = OptixPixelFormat::OPTIX_PIXEL_FORMAT_FLOAT4;
	//		state.denoiserInputs[2].width = state.lightmappingParams.imageSize.x;
	//		state.denoiserInputs[2].height = state.lightmappingParams.imageSize.y;
	//		state.denoiserInputs[2].pixelStrideInBytes = sizeof(float4);
	//		state.denoiserInputs[2].rowStrideInBytes = state.lightmappingParams.imageSize.x * sizeof(float4);

	//		state.denoiserOutput = state.denoiserInputs[0];
	//		state.denoiserOutput.data = state.denoisedColorBuffer.data;
	//	}

	//	// Setup denoiser
	//	if (setup)
	//	{
	//		OPTIX_CHECK(optixDenoiserSetup(
	//			state.denoiser,
	//			0,  // CUDA stream
	//			state.lightmappingParams.imageSize.x,
	//			state.lightmappingParams.imageSize.y,
	//			state.denoiserState.data,
	//			state.denoiserState.sizeInBytes,
	//			state.denoiserScratch.data,
	//			state.denoiserScratch.sizeInBytes
	//		));

	//		state.denoiserParams.denoiseAlpha = 0;
	//		state.denoiserParams.hdrIntensity = state.denoiserIntensity.data;
	//		state.denoiserParams.blendFactor = 0.0f;
	//	}
	//}

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
			if (s_ParamsPtr == 0)
			{
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_ParamsPtr), sizeof(LightmappingParams)));
			}

			CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void*>(s_ParamsPtr),
				&state.lightmappingParams, sizeof(LightmappingParams),
				cudaMemcpyHostToDevice
			));

			OPTIX_CHECK(optixLaunch(state.pipeline, state.stream, s_ParamsPtr, sizeof(LightmappingParams), &state.sbt, state.launchSize.x, state.launchSize.y, 1));
			cudaError_t err = cudaGetLastError();
			CUDA_SYNC_CHECK();
		}
	}

	//void LaunchDenoiser(LightmapperState& state)
	//{
	//	// Launch denoiser
	//	{
	//		OPTIX_CHECK(optixDenoiserComputeIntensity(
	//			state.denoiser,
	//			0, // CUDA stream
	//			&state.denoiserInputs[0],
	//			state.denoiserIntensity.data,
	//			state.denoiserScratch.data,
	//			state.denoiserScratch.sizeInBytes
	//		));

	//		OPTIX_CHECK(optixDenoiserInvoke(
	//			state.denoiser,
	//			0, // CUDA stream
	//			&state.denoiserParams,
	//			state.denoiserState.data,
	//			state.denoiserState.sizeInBytes,
	//			state.denoiserInputs,
	//			3, // num input channels
	//			0, // input offset X
	//			0, // input offset y
	//			&state.denoiserOutput,
	//			state.denoiserScratch.data,
	//			state.denoiserScratch.sizeInBytes
	//		));

	//		CUDA_SYNC_CHECK();
	//	}
	//}

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

			for (uint32_t i = 0; i < 4; ++i)
			{
				CUDA_CHECK_RESULT(cuMemcpyHtoD(s_DenoisingParamsPtr, &state.denoisingParams, sizeof(DenoisingParams)));
				CUDA_CHECK_RESULT(cuLaunchKernel(state.denoiserKernel, state.lightmappingParams.imageSize.x / 8, state.lightmappingParams.imageSize.y / 8, 1, 8, 8, 1, 0, 0, 0, 0));
				CUDA_CHECK_RESULT(cuCtxSynchronize());
				
				float4* input = state.denoisingParams.inputColor;
				state.denoisingParams.inputColor = state.denoisingParams.outputColor;
				state.denoisingParams.outputColor = input;
			}
		}
	}

	void LightmappingManager::Clear()
	{
		if (s_State.context == nullptr)
		{
			return;
		}
		s_State.lightmappingParams.accumulationFrameIndex = 0;
		s_State.accumulatedFrameBuffer.clear();
	}

	void LightmappingManager::Calculate(Scene* scene, Camera* camera, const Vector2Int& viewport, uint8_t* output)
	{
		if (s_State.context == nullptr)
		{
			CreateContext(s_State);
			//CreateDenoiser(s_State);
			InitializeAccels(s_State, scene);
			CreatePTXModule(s_State);
			CreateProgramGroups(s_State);
			CreatePipeline(s_State);
			CreateSBT(s_State);
			InitializeParams(s_State);
		}

		if (s_State.lightmappingParams.accumulationFrameIndex < ACCUMULATION_FRAMES_COUNT - 1)
		{
			InitializeFrameBufferAndCamera(s_State, camera, viewport);
			//InitializeDenoiserFrameBuffer(s_State);
			InitializeLights(s_State, scene);
			Launch(s_State);
			if (s_State.lightmappingParams.accumulationFrameIndex == ACCUMULATION_FRAMES_COUNT - 1)
			{
				//LaunchDenoiser(s_State);
			}
			s_State.colorBuffer.download(output, s_State.colorBuffer.sizeInBytes);
		}
		else
		{
			s_State.denoisedColorBuffer.download(output, s_State.denoisedColorBuffer.sizeInBytes);
		}
	}

	void LightmappingManager::Calculate(Scene* scene, const Vector2Int& tileSize, uint8_t*& output, Vector2Int& outputSize, List<Vector4>& chartScaleOffset, Dictionary<ObjectId, uint32_t>& chartInstanceOffset)
	{
		s_State = {};
		CreateContext(s_State);
		//CreateDenoiser(s_State);
		CreateDenoiserPTXModule(s_State);
		InitializeAccels(s_State, scene);
		CreatePTXModule(s_State);
		CreateProgramGroups(s_State);
		CreatePipeline(s_State);
		CreateSBT(s_State);
		InitializeParams(s_State);

		outputSize = Vector2Int(s_Size, s_Size);
		Vector4* temp = BB_MALLOC_ARRAY(Vector4, s_Size * s_Size);
		Vector4* result = BB_MALLOC_ARRAY(Vector4, s_Size * s_Size);
		output = reinterpret_cast<uint8_t*>(result);
		InitializeFrameBuffer(s_State, outputSize);
		//InitializeDenoiserFrameBuffer(s_State);
		InitializeLights(s_State, scene);

		InitializeChartsAndBVH(s_State);

		//uint32_t size = 1024;
		//outputSize = Vector2Int(size, size);
		//output = BB_MALLOC_ARRAY(uint8_t, size * size * 4);

		/*static Vector3Int chartColors[] = { Vector3Int(255, 255, 255), Vector3Int(0, 255, 255), Vector3Int(255, 0, 255), Vector3Int(255, 255, 0), Vector3Int(0, 0, 255), Vector3Int(255, 0, 0), Vector3Int(0, 255, 0) };
		for (uint32_t i = 0; i < size; ++i)
		{
			for (uint32_t j = 0; j < size; ++j)
			{
				uint32_t index = j * size + i;
				uint32_t chartIndex = s_State.atlasMask[index];
				if (chartIndex != 0)
				{
					uint8_t* ptr = output + index * 4;
					Vector3Int color = chartColors[chartIndex % ARRAYSIZE(chartColors)];
					*ptr = color.x;
					*(ptr + 1) = color.y;
					*(ptr + 2) = color.z;
					*(ptr + 3) = 255;
				}
			}
		}*/
		Vector2Int launchTileSize = Vector2Int(std::min(static_cast<int>(s_Size), tileSize.x), std::min(static_cast<int>(s_Size), tileSize.y));

		s_State.launchSize = make_uint2(launchTileSize.x, launchTileSize.y);
		uint32_t tilesX = outputSize.x / launchTileSize.x;
		uint32_t tilesY = outputSize.y / launchTileSize.y;
		for (uint32_t i = 0; i < tilesX; ++i)
		{
			for (uint32_t j = 0; j < tilesY; ++j)
			{
				s_State.lightmappingParams.offset = make_uint2(i * launchTileSize.x, j * launchTileSize.y);
				Launch(s_State);
				BB_INFO(i << " " << j);
			}
		}
		LaunchDenoiser(s_State);

		s_State.denoisedColorBuffer.download(temp, s_State.denoisedColorBuffer.sizeInBytes / sizeof(Vector4));

		int atlasSize = static_cast<int>(s_State.atlasMask.size());
		for (int j = 0; j < outputSize.y; ++j)
		{
			for (int i = 0; i < outputSize.x; ++i)
			{
				int index = j * outputSize.x + i;
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
						int nearbyIndex = (j + offset.y) * outputSize.x + (i + offset.x);
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

		BB_FREE(temp);
		chartScaleOffset = s_State.chartOffsetScale;
		chartInstanceOffset = s_State.chartInstanceOffset;
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
			s_State.accumulatedFrameBuffer.free();
			s_State.colorBuffer.free();
			s_State.albedoBuffer.free();
			s_State.normalBuffer.free();
			s_State.denoisedColorBuffer.free();
			//s_State.denoiserIntensity.free();
			//s_State.denoiserScratch.free();
			//s_State.denoiserState.free();

			OPTIX_CHECK(optixDeviceContextDestroy(s_State.context));
			cuModuleUnload(s_State.denoiserPtxModule);
			s_State = {};
		}
	}
}
