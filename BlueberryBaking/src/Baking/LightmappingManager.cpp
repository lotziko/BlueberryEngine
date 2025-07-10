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
#include "Params.h"
#include "VecMath.h"

#include <iomanip>
#include <iostream>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

namespace Blueberry
{
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

		OptixDenoiser denoiser = nullptr;
		OptixDenoiserParams denoiserParams = {};
		OptixImage2D denoiserInput = {};
		OptixImage2D denoiserOutput = {};
		CUDABuffer denoiserIntensity = {};
		CUDABuffer denoiserScratch = {};
		CUDABuffer denoiserState = {};

		Dictionary<ObjectId, MeshData> meshDatas = {};
		OptixTraversableHandle iasHandle = 0;

		OptixPipelineCompileOptions pipelineCompileOptions = {};
		OptixPipeline pipeline = nullptr;
		OptixModule ptxModule = {};

		OptixProgramGroup raygenProgGroup = 0;
		OptixProgramGroup missProgGroup = 0;
		OptixProgramGroup hitgroupDefaultProgGroup = 0;
		OptixProgramGroup hitgroupShadowProgGroup = 0;

		CUstream stream = 0;
		CUDABuffer instanceMatrices = {};
		CUDABuffer accumulatedFrameBuffer = {};
		CUDABuffer frameBuffer = {};
		CUDABuffer denoisedFrameBuffer = {};
		Params params = {};
		
		OptixShaderBindingTable sbt = {};
	} s_State = {};

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

	void CreateDenoiser(LightmapperState& state)
	{
		// Create denoiser
		{
			OptixDenoiserOptions options = {};
			options.inputKind = OPTIX_DENOISER_INPUT_RGB;
			OPTIX_CHECK(optixDenoiserCreate(state.context, &options, &state.denoiser));
			OPTIX_CHECK(optixDenoiserSetModel(
				state.denoiser,
				OPTIX_DENOISER_MODEL_KIND_HDR,
				nullptr, // data
				0        // size
			));
		}
	}

	void InitializeAccels(LightmapperState& state, Scene* scene)
	{
		// Initialize meshes
		{
			for (auto component : scene->GetIterator<MeshRenderer>())
			{
				MeshRenderer* meshRenderer = static_cast<MeshRenderer*>(component.second);
				Mesh* mesh = meshRenderer->GetMesh();
				auto it = state.meshDatas.find(mesh->GetObjectId());
				if (it != state.meshDatas.end())
				{
					it->second.transforms.emplace_back(meshRenderer->GetTransform());
				}
				else
				{
					MeshData data = {};
					data.mesh = mesh;
					data.transforms.emplace_back(meshRenderer->GetTransform());
					data.vertexBuffer.alloc_and_upload(mesh->GetVertices().data(), mesh->GetVertexCount());
					data.normalBuffer.alloc_and_upload(mesh->GetNormals().data(), mesh->GetVertexCount());
					data.tangentBuffer.alloc_and_upload(mesh->GetTangents().data(), mesh->GetVertexCount());
					data.indexBuffer.alloc_and_upload(mesh->GetIndices().data(), mesh->GetIndexCount());
					state.meshDatas.insert({ mesh->GetObjectId(), data });
				}
			}
		}

		// Build acceleration structures
		{
			OptixAccelBuildOptions accelOptions = {};
			accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
			accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

			for (auto& pair : state.meshDatas)
			{
				Mesh* mesh = pair.second.mesh;
				OptixBuildInput buildInput = {};
				unsigned int flags = 1;

				buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
				buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
				buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
				buildInput.triangleArray.numVertices = static_cast<unsigned int>(mesh->GetVertexCount());
				buildInput.triangleArray.vertexBuffers = &pair.second.vertexBuffer.data;
				buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
				buildInput.triangleArray.indexStrideInBytes = sizeof(int3);
				buildInput.triangleArray.numIndexTriplets = static_cast<unsigned int>(mesh->GetIndexCount()) / 3;
				buildInput.triangleArray.indexBuffer = pair.second.indexBuffer.data;
				buildInput.triangleArray.flags = &flags;
				buildInput.triangleArray.numSbtRecords = 1;

				OptixAccelBufferSizes gasBufferSizes;
				OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &accelOptions, &buildInput, 1, &gasBufferSizes));

				CUDA_CHECK(cudaMalloc((void**)&pair.second.gasOutput, gasBufferSizes.outputSizeInBytes));

				CUdeviceptr tempBuffer;
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tempBuffer), gasBufferSizes.tempSizeInBytes));

				OPTIX_CHECK(optixAccelBuild(state.context, 0,   // CUDA stream
					&accelOptions,
					&buildInput,
					1,
					tempBuffer,
					gasBufferSizes.tempSizeInBytes,
					pair.second.gasOutput,
					gasBufferSizes.outputSizeInBytes,
					&pair.second.gasHandle,
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
			for (auto& pair : state.meshDatas)
			{
				instanceCount += pair.second.transforms.size();
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

			for (auto& pair : state.meshDatas)
			{
				for (Transform* transform : pair.second.transforms)
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
					optixInstance.traversableHandle = pair.second.gasHandle;

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
			state.pipelineCompileOptions.numPayloadValues = 5;
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
			for (auto& pair : state.meshDatas)
			{
				HitGroupSbtRecord defaultRecord = {};
				defaultRecord.data.vertices = (float3*)pair.second.vertexBuffer.data;
				defaultRecord.data.normals = (float3*)pair.second.normalBuffer.data;
				defaultRecord.data.tangents = (float4*)pair.second.tangentBuffer.data;
				defaultRecord.data.indices = (uint3*)pair.second.indexBuffer.data;
				OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroupDefaultProgGroup, &defaultRecord));
				hitgroupRecordsList.push_back(defaultRecord);

				HitGroupSbtRecord shadowRecord = {};
				shadowRecord.data.vertices = (float3*)pair.second.vertexBuffer.data;
				shadowRecord.data.normals = (float3*)pair.second.normalBuffer.data;
				shadowRecord.data.tangents = (float4*)pair.second.tangentBuffer.data;
				shadowRecord.data.indices = (uint3*)pair.second.indexBuffer.data;
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
		state.params.handle = state.iasHandle;
		state.params.instanceMatrices = reinterpret_cast<Matrix3x4*>(state.instanceMatrices.data);
	}

	void InitializeFrameBufferAndCamera(LightmapperState& state, Camera* camera, const Vector2Int& viewport)
	{
		state.params.accumulationFrameIndex += 1;
		state.params.imageWidth = viewport.x;
		state.params.imageHeight = viewport.y;
		GetCameraParams(camera, state.params.camEye, state.params.camU, state.params.camV, state.params.camW);

		size_t accumulatedFrameBufferSize = state.params.imageWidth * state.params.imageHeight * sizeof(float4);
		size_t frameBufferSize = state.params.imageWidth * state.params.imageHeight * sizeof(float4);
		if (state.params.image == nullptr)
		{
			state.accumulatedFrameBuffer.alloc(accumulatedFrameBufferSize);
			state.params.accumulatedImage = reinterpret_cast<float4*>(state.accumulatedFrameBuffer.data);
			state.frameBuffer.alloc(frameBufferSize);
			state.denoisedFrameBuffer.alloc(frameBufferSize);
			state.params.image = reinterpret_cast<float4*>(state.frameBuffer.data);
		}
		else if (state.frameBuffer.sizeInBytes != frameBufferSize)
		{
			state.accumulatedFrameBuffer.resize(accumulatedFrameBufferSize);
			state.frameBuffer.resize(frameBufferSize);
			state.denoisedFrameBuffer.resize(frameBufferSize);
		}
	}

	void InitializeDenoiserFrameBuffer(LightmapperState& state)
	{
		// Allocate memory
		bool setup = false;
		{
			OptixDenoiserSizes denoiserSizes;
			OPTIX_CHECK(optixDenoiserComputeMemoryResources(
				state.denoiser,
				state.params.imageWidth,
				state.params.imageHeight,
				&denoiserSizes
			));

			if (state.denoiserScratch.sizeInBytes == 0)
			{
				setup = true;
				state.denoiserIntensity.alloc(sizeof(float));
				state.denoiserScratch.alloc(denoiserSizes.withoutOverlapScratchSizeInBytes);
				state.denoiserState.alloc(denoiserSizes.stateSizeInBytes);
			}
			else if (state.denoiserScratch.sizeInBytes != denoiserSizes.withoutOverlapScratchSizeInBytes)
			{
				state.denoiserScratch.resize(denoiserSizes.withoutOverlapScratchSizeInBytes);
				state.denoiserState.resize(denoiserSizes.stateSizeInBytes);
			}
		}

		// Setup images
		{
			state.denoiserInput.data = state.frameBuffer.data;
			state.denoiserInput.format = OptixPixelFormat::OPTIX_PIXEL_FORMAT_FLOAT4;
			state.denoiserInput.width = state.params.imageWidth;
			state.denoiserInput.height = state.params.imageHeight;
			state.denoiserInput.pixelStrideInBytes = sizeof(float4);
			state.denoiserInput.rowStrideInBytes = state.params.imageWidth * sizeof(float4);

			state.denoiserOutput = state.denoiserInput;
			state.denoiserOutput.data = state.denoisedFrameBuffer.data;
		}

		// Setup denoiser
		if (setup)
		{
			OPTIX_CHECK(optixDenoiserSetup(
				state.denoiser,
				0,  // CUDA stream
				state.params.imageWidth,
				state.params.imageHeight,
				state.denoiserState.data,
				state.denoiserState.sizeInBytes,
				state.denoiserScratch.data,
				state.denoiserScratch.sizeInBytes
			));

			state.denoiserParams.denoiseAlpha = 0;
			state.denoiserParams.hdrIntensity = state.denoiserIntensity.data;
			state.denoiserParams.blendFactor = 0.0f;
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
				state.params.directionalLight.direction = { dir.x, dir.y, dir.z };
				state.params.directionalLight.color = { color.R() * intensity, color.G() * intensity, color.B() * intensity };
			}
		}
	}

	void Launch(LightmapperState& state)
	{
		// Launch
		{
			CUdeviceptr paramsPtr;

			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&paramsPtr), sizeof(Params)));
			CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void*>(paramsPtr),
				&state.params, sizeof(Params),
				cudaMemcpyHostToDevice
			));

			OPTIX_CHECK(optixLaunch(state.pipeline, state.stream, paramsPtr, sizeof(Params), &state.sbt, state.params.imageWidth, state.params.imageHeight, 1));
			cudaError_t err = cudaGetLastError();
			CUDA_SYNC_CHECK();
		}
	}

	void LaunchDenoiser(LightmapperState& state)
	{
		// Launch denoiser
		{
			OPTIX_CHECK(optixDenoiserComputeIntensity(
				state.denoiser,
				0, // CUDA stream
				&state.denoiserInput,
				state.denoiserIntensity.data,
				state.denoiserScratch.data,
				state.denoiserScratch.sizeInBytes
			));

			OPTIX_CHECK(optixDenoiserInvoke(
				state.denoiser,
				0, // CUDA stream
				&state.denoiserParams,
				state.denoiserState.data,
				state.denoiserState.sizeInBytes,
				&state.denoiserInput,
				1, // num input channels
				0, // input offset X
				0, // input offset y
				&state.denoiserOutput,
				state.denoiserScratch.data,
				state.denoiserScratch.sizeInBytes
			));

			CUDA_SYNC_CHECK();
		}
	}

	void LightmappingManager::Clear()
	{
		if (s_State.context == nullptr)
		{
			return;
		}
		s_State.params.accumulationFrameIndex = 0;
		s_State.accumulatedFrameBuffer.clear();
	}

	void LightmappingManager::Calculate(Scene* scene, Camera* camera, const Vector2Int& viewport, uint8_t* output)
	{
		if (s_State.context == nullptr)
		{
			CreateContext(s_State);
			CreateDenoiser(s_State);
			InitializeAccels(s_State, scene);
			CreatePTXModule(s_State);
			CreateProgramGroups(s_State);
			CreatePipeline(s_State);
			CreateSBT(s_State);
			InitializeParams(s_State);
		}

		if (s_State.params.accumulationFrameIndex < ACCUMULATION_FRAMES_COUNT - 1)
		{
			InitializeFrameBufferAndCamera(s_State, camera, viewport);
			InitializeDenoiserFrameBuffer(s_State);
			InitializeLights(s_State, scene);
			Launch(s_State);
			if (s_State.params.accumulationFrameIndex == ACCUMULATION_FRAMES_COUNT - 1)
			{
				LaunchDenoiser(s_State);
			}
			s_State.frameBuffer.download(output, s_State.frameBuffer.sizeInBytes);
		}
		else
		{
			s_State.denoisedFrameBuffer.download(output, s_State.denoisedFrameBuffer.sizeInBytes);
		}
	}

	void LightmappingManager::Shutdown()
	{
		// Release denoiser
		{
			optixDenoiserDestroy(s_State.denoiser);
		}

		// Release context
		{
			for (auto& pair : s_State.meshDatas)
			{
				pair.second.Release();
			}
			s_State.instanceMatrices.free();
			s_State.accumulatedFrameBuffer.free();
			s_State.frameBuffer.free();
			s_State.denoisedFrameBuffer.free();
			s_State.denoiserIntensity.free();
			s_State.denoiserScratch.free();
			s_State.denoiserState.free();

			OPTIX_CHECK(optixDeviceContextDestroy(s_State.context));
			s_State = {};
		}
	}
}
