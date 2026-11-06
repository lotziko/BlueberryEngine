#include "Baking\LightmappingManager.h"

#include "Blueberry\Core\Time.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Scene\Components\SkyRenderer.h"
#include "Blueberry\Scene\Components\Light.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\GfxTexturePool.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\DefaultMaterials.h"
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
	struct LightmappingChartData
	{
		List<uint64_t> mask;
		Vector2Int maskSize;
		Vector4 meshBounds;
		Vector4 maskBounds;
		Vector2Int position;
		Vector2Int size;
		uint32_t index;
	};

	struct LightmappingInstanceData
	{
		Mesh* mesh;
		MeshRenderer* renderer;
		Transform* transform;
	};

	struct LightmappingMeshData
	{
		void Release()
		{
			vertexBuffer.free();
			normalBuffer.free();
			tangentBuffer.free();
			uvBuffer.free();
			submeshBuffer.free();
			indexBuffer.free();
			CUDA_CHECK(cudaFree((void*)gasOutput));
		}

		uint32_t updateCount;
		List<Vector4> chartBounds;

		CUDABuffer vertexBuffer;
		CUDABuffer normalBuffer;
		CUDABuffer tangentBuffer;
		CUDABuffer uvBuffer;
		CUDABuffer submeshBuffer;
		CUDABuffer indexBuffer;

		OptixTraversableHandle gasHandle;
		CUdeviceptr gasOutput;
	};

	struct LightmappingTextureData
	{
		uint32_t updateCount;
		CUDABuffer buffer;
	};

	template <typename T>
	struct Record
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		T data;
	};

	#define CLOSEST_HIT_COUNT 3

	struct LightmappingPassState
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
		bool isInitialized = false;
		bool hasSky = false;
		OptixDeviceContext context = nullptr;
		CalculationParams params = {};

		Dictionary<ObjectId, LightmappingMeshData> meshDatas = {};
		Dictionary<ObjectId, LightmappingTextureData> textureDatas = {};
		List<LightmappingInstanceData> instanceDatas = {};
		CUdeviceptr iasOutput = 0;
		OptixTraversableHandle iasHandle = 0;

		LightmappingPassState pass = {};

		CUmodule denoiserPtxModule = {};
		CUfunction denoiserKernelFirstPass = {};
		CUfunction denoiserKernelpass = {};

		List<uint2> validTexels = {};
		List<uint64_t> atlasMask = {};
		List<uint32_t> chartIndexMask = {};
		CalculationResult result = {};
		CUDABuffer pointLights = {};
		CUDABVH bvh = {};
		
		unsigned int* completeTexelCounter;
		unsigned int* completeProbeCounter;

		CUstream stream = 0;
		CUDABuffer instanceMatrices = {};
		CUDABuffer validTexelsBuffer = {};
		CUDABuffer colorBuffer = {};
		CUDABuffer normalBuffer = {};
		CUDABuffer positionBuffer = {};
		CUDABuffer chartIndexBuffer = {};
		CUDABuffer denoisedColorBuffer = {};
		CUDABuffer probePositionBuffer = {};
		CUDABuffer probeColorBuffer = {};
		LightmappingParams lightmappingParams = {};
		DenoisingParams denoisingParams = {};
	} s_State = {};

	static CUdeviceptr s_ParamsPtr = 0;
	static CUdeviceptr s_DenoisingParamsPtr = 0;
	static int s_Padding = 0;
	static size_t s_BaseMapId = TO_HASH("_BaseMap");

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
		
		// Create stream
		{
			CUDA_CHECK(cudaStreamCreate(&state.stream));
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
				state.pass.pipelineCompileOptions.numPayloadValues = 8;
				state.pass.pipelineCompileOptions.numAttributeValues = 2;
				state.pass.pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE; // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
				state.pass.pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

				String ptx = FileHelper::LoadText("assets\\ptx\\Lightmapping.ptx");
				
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
		// Create denoiser PTX module
		{
			CUDA_CHECK_RESULT(cuModuleLoad(&state.denoiserPtxModule, "assets\\ptx\\Denoising.ptx"));
			CUDA_CHECK_RESULT(cuModuleGetFunction(&state.denoiserKernelFirstPass, state.denoiserPtxModule, "__denoise__firstpass"));
			CUDA_CHECK_RESULT(cuModuleGetFunction(&state.denoiserKernelpass, state.denoiserPtxModule, "__denoise__secondpass"));
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

	void InitializeState(LightmapperState& state)
	{
		if (state.isInitialized)
		{
			return;
		}

		CreateContext(state);
		CreatePTXModule(state);
		CreateDenoiserPTXModule(state);
		CreateProgramGroups(state);
		CreatePipeline(state);

		state.isInitialized = true;
	}

	void BuildTextureData(GfxTexture* renderTexture, Texture* texture, bool isTextureCube, LightmappingTextureData& data)
	{
		if (isTextureCube)
		{
			GfxDevice::SetViewCount(6);
			GfxDevice::SetRenderTarget(renderTexture);
			GfxDevice::SetViewport(0, 0, SKY_COLOR_SIZE, SKY_COLOR_SIZE);
			GfxDevice::SetGlobalTexture(TO_HASH("_BlitTexture"), texture->Get());
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), DefaultMaterials::GetBlit(), 1)); // TODO render material instead of blit
			GfxDevice::SetViewCount(1);
			GfxDevice::SetRenderTarget(nullptr);

			List<uint8_t> temporarybuffer = {};
			temporarybuffer.resize(SKY_COLOR_SIZE * SKY_COLOR_SIZE * 6 * 4);
			List<uint8_t> buffer = {};
			buffer.resize(temporarybuffer.size());
			renderTexture->GetData(temporarybuffer.data());

			uint32_t bytesPerPixel = 4;
			size_t rowPitch = SKY_COLOR_SIZE * bytesPerPixel;
			size_t wideRowPitch = rowPitch * 6;
			size_t slicePitch = rowPitch * SKY_COLOR_SIZE;
			size_t wideSlicePitch = slicePitch * 6;
			for (uint32_t i = 0; i < 6; ++i)
			{
				for (uint32_t j = 0; j < SKY_COLOR_SIZE; ++j)
				{
					memcpy(buffer.data() + i * rowPitch + j * wideRowPitch, temporarybuffer.data() + i * slicePitch + j * rowPitch, SKY_COLOR_SIZE * bytesPerPixel);
				}
			}
			data.buffer.alloc_and_upload(buffer.data(), buffer.size());
		}
		else
		{
			GfxDevice::SetRenderTarget(renderTexture);
			GfxDevice::SetViewport(0, 0, MATERIAL_COLOR_SIZE, MATERIAL_COLOR_SIZE);
			GfxDevice::SetGlobalTexture(TO_HASH("_BlitTexture"), texture->Get());
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), DefaultMaterials::GetBlit(), 0));
			GfxDevice::SetRenderTarget(nullptr);
			List<uint8_t> buffer = {};
			buffer.resize(MATERIAL_COLOR_SIZE * MATERIAL_COLOR_SIZE * 4);
			renderTexture->GetData(buffer.data());
			data.buffer.alloc_and_upload(buffer.data(), buffer.size());
		}
	}

	void CreateTextures(LightmapperState& state, Scene* scene)
	{
		GfxTexture* renderTexture = GfxTexturePool::Get(MATERIAL_COLOR_SIZE, MATERIAL_COLOR_SIZE, 1, TextureUsageFlags::RenderTarget | TextureUsageFlags::CPUReadable, 1, 1, TextureFormat::R8G8B8A8_UNorm, TextureDimension::Texture2D, WrapMode::Clamp, FilterMode::Bilinear);
		for (MeshRenderer* meshRenderer : scene->GetComponents<MeshRenderer>())
		{
			for (uint32_t i = 0; i < meshRenderer->GetMaterialCount(); ++i)
			{
				Material* material = meshRenderer->GetMaterial(i);
				if (material == nullptr)
				{
					continue;
				}
				Texture* texture = material->GetTexture(s_BaseMapId);
				if (texture == nullptr)
				{
					continue;
				}
				ObjectId objectId = texture->GetObjectId();
				auto it = state.textureDatas.find(objectId);
				bool needsTextureBuild = false;
				if (it == state.textureDatas.end())
				{
					needsTextureBuild = true;
				}
				else if (it->second.updateCount != texture->GetUpdateCount())
				{
					it->second.buffer.free();
					needsTextureBuild = true;
				}
				if (needsTextureBuild)
				{
					LightmappingTextureData textureData = {};
					textureData.updateCount = texture->GetUpdateCount();
					BuildTextureData(renderTexture, texture, false, textureData);
					state.textureDatas.insert_or_assign(objectId, std::move(textureData));
				}
			}
		}
		GfxTexturePool::Release(renderTexture);

		state.hasSky = false;
		for (SkyRenderer* skyRenderer : scene->GetComponents<SkyRenderer>())
		{
			Material* material = skyRenderer->GetMaterial();
			if (material == nullptr)
			{
				continue;
			}
			Texture* texture = material->GetTexture(s_BaseMapId);
			if (texture == nullptr)
			{
				continue;
			}
			auto it = state.textureDatas.find(INVALID_ID);
			bool needsTextureBuild = false;
			if (it == state.textureDatas.end())
			{
				needsTextureBuild = true;
			}
			else if (it->second.updateCount != texture->GetUpdateCount())
			{
				it->second.buffer.free();
				needsTextureBuild = true;
			}
			if (needsTextureBuild)
			{
				GfxTexture* skyRenderTexture = GfxTexturePool::Get(SKY_COLOR_SIZE, SKY_COLOR_SIZE, 1, TextureUsageFlags::RenderTarget | TextureUsageFlags::CPUReadable, 1, 1, TextureFormat::R8G8B8A8_UNorm, TextureDimension::TextureCube, WrapMode::Clamp, FilterMode::Bilinear);
				LightmappingTextureData textureData = {};
				BuildTextureData(skyRenderTexture, texture, true, textureData);
				state.textureDatas.insert_or_assign(INVALID_ID, std::move(textureData));
				GfxTexturePool::Release(skyRenderTexture);
			}
			state.hasSky = true;
			break;
		}
	}

	void BuildMeshData(LightmapperState& state, Mesh* mesh, LightmappingMeshData& data)
	{
		data.updateCount = mesh->GetUpdateCount();
		data.vertexBuffer.alloc_and_upload(mesh->GetVertices(), mesh->GetVertexCount());
		data.normalBuffer.alloc_and_upload(mesh->GetNormals(), mesh->GetVertexCount());
		data.tangentBuffer.alloc_and_upload(mesh->GetTangents(), mesh->GetVertexCount());
		data.uvBuffer.alloc_and_upload(mesh->GetUVs(0), mesh->GetVertexCount() * 2);
		data.indexBuffer.alloc_and_upload(mesh->GetIndices(), mesh->GetIndexCount());

		uint32_t submeshCount = mesh->GetSubMeshCount();
		if (submeshCount > 1)
		{
			List<uint32_t> submeshOffsets = {};
			submeshOffsets.resize(mesh->GetIndexCount());
			for (uint32_t i = 0; i < submeshCount; ++i)
			{
				auto& submesh = mesh->GetSubMesh(i);
				uint32_t* begin = submeshOffsets.data() + submesh.GetIndexStart();
				uint32_t* end = begin + submesh.GetIndexCount();
				std::fill(begin, end, i);
			}
			data.submeshBuffer.alloc_and_upload(submeshOffsets.data(), mesh->GetIndexCount());
		}

		OptixAccelBuildOptions accelOptions = {};
		accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
		accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		OptixBuildInput buildInput = {};
		List<unsigned int> flags = {};
		flags.resize(mesh->GetSubMeshCount());
		std::fill(flags.begin(), flags.end(), 1); // any hit disabled

		buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
		buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
		buildInput.triangleArray.numVertices = static_cast<unsigned int>(mesh->GetVertexCount());
		buildInput.triangleArray.vertexBuffers = &data.vertexBuffer.data;
		buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		buildInput.triangleArray.indexStrideInBytes = sizeof(int3);
		buildInput.triangleArray.numIndexTriplets = static_cast<unsigned int>(mesh->GetIndexCount()) / 3;
		buildInput.triangleArray.indexBuffer = data.indexBuffer.data;
		buildInput.triangleArray.flags = flags.data();
		buildInput.triangleArray.numSbtRecords = mesh->GetSubMeshCount();
		buildInput.triangleArray.sbtIndexOffsetBuffer = data.submeshBuffer.data;
		buildInput.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
		buildInput.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

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

		Vector3* uvs = reinterpret_cast<Vector3*>(mesh->GetUVs(1));
		if (uvs != nullptr)
		{
			float maxChart = 0;
			for (uint32_t i = 0; i < mesh->GetVertexCount(); ++i)
			{
				maxChart = std::max(uvs[i].z, maxChart);
			}
			uint32_t chartCount = static_cast<uint32_t>(maxChart + 1);
			for (uint32_t i = 0; i < chartCount; ++i)
			{
				float chartIndex = static_cast<float>(i);
				Vector4 chartBounds = { FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX };
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
				if (chartBounds.x != FLT_MAX)
				{
					data.chartBounds.push_back(chartBounds);
				}
				else
				{
					data.chartBounds.push_back(Vector4::Zero);
				}
			}
		}
	}

	void CreateAccelerationStructures(LightmapperState& state, Scene* scene)
	{
		state.instanceDatas.clear();

		// Build acceleration structures
		{
			for (MeshRenderer* meshRenderer : scene->GetComponents<MeshRenderer>())
			{
				Transform* transform = meshRenderer->GetTransform();
				if (!transform->IsStatic())
				{
					continue;
				}
				Mesh* mesh = meshRenderer->GetMesh();
				if (mesh == nullptr)
				{
					BB_ERROR(meshRenderer->GetEntity()->GetName() << " has no mesh.");
					continue;
				}
				VertexLayout layout = mesh->GetLayout();
				if (meshRenderer->GetMaterialCount() == 0)
				{
					BB_ERROR(meshRenderer->GetEntity()->GetName() << " has no materials.");
					continue;
				}
				if (!layout.Has(VertexAttribute::Tangent))
				{
					BB_ERROR(mesh->GetName() << " has no tangents.");
					continue;
				}
				if (state.result.hasLightmap && meshRenderer->IsBakeable() && !layout.Has(VertexAttribute::Texcoord1))
				{
					BB_ERROR(mesh->GetName() << " has no lightmap UVs.");
				}

				ObjectId meshId = mesh->GetObjectId();
				auto it = state.meshDatas.find(meshId);
				bool needsMeshBuild = false;
				if (it == state.meshDatas.end())
				{
					needsMeshBuild = true;
				}
				else if (it->second.updateCount != mesh->GetUpdateCount())
				{
					it->second.Release();
					needsMeshBuild = true;
				}
				if (needsMeshBuild)
				{
					LightmappingMeshData meshData = {};
					BuildMeshData(state, mesh, meshData);
					state.meshDatas.insert_or_assign(meshId, std::move(meshData));
				}
				LightmappingInstanceData instanceData = {};
				instanceData.mesh = mesh;
				instanceData.renderer = meshRenderer;
				instanceData.transform = transform;
				state.instanceDatas.push_back(std::move(instanceData));
			}
		}

		// Build instance acceleration structures
		{
			if (state.iasOutput)
			{
				CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.iasOutput)));
			}

			uint32_t instanceCount = static_cast<uint32_t>(state.instanceDatas.size());
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
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.iasOutput), iasBufferSizes.outputSizeInBytes));

			OptixInstance* optixInstanceArray = BB_MALLOC_ARRAY(OptixInstance, instanceCount);
			Matrix3x4* instanceMatrixArray = BB_MALLOC_ARRAY(Matrix3x4, instanceCount);
			size_t instanceOffset = 0;
			size_t sbtOffset = 0;

			for (auto& instanceData : state.instanceDatas)
			{
				size_t sbtCount = instanceData.mesh->GetSubMeshCount() * CLOSEST_HIT_COUNT;

				Matrix transformMatrix = instanceData.transform->GetLocalToWorldMatrix();
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
				optixInstance.sbtOffset = static_cast<unsigned int>(sbtOffset);
				optixInstance.visibilityMask = 1u;
				optixInstance.traversableHandle = state.meshDatas[instanceData.mesh->GetObjectId()].gasHandle;

				optixInstanceArray[instanceOffset] = optixInstance;
				instanceMatrixArray[instanceOffset] = Matrix3x4(tr);

				instanceOffset += 1;
				sbtOffset += sbtCount;
			}
			
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(instanceBuffer), optixInstanceArray, instanceSizeInBytes,
				cudaMemcpyHostToDevice));
			state.instanceMatrices.resize_and_upload(instanceMatrixArray, instanceCount);
			state.lightmappingParams.instanceMatrices = reinterpret_cast<Matrix3x4*>(state.instanceMatrices.data);
			BB_FREE(optixInstanceArray);
			BB_FREE(instanceMatrixArray);

			OPTIX_CHECK(optixAccelBuild(state.context,
				0,  // CUDA stream
				&accelOptions,
				&instanceInput,
				1,  // num build inputs
				tempBuffer,
				iasBufferSizes.tempSizeInBytes,
				state.iasOutput,
				iasBufferSizes.outputSizeInBytes,
				&state.iasHandle,
				nullptr,  // emitted property list
				0u         // num emitted properties
			));
			state.lightmappingParams.handle = state.iasHandle;

			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(tempBuffer)));
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(instanceBuffer)));
		}
	}

	void CreateSBT(LightmapperState& state)
	{
		// Set up shader binding table
		{
			// Pass
			{
				// Clear
				if (state.pass.sbt.raygenRecord)
				{
					CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.pass.sbt.raygenRecord)));
				}
				if (state.pass.sbt.missRecordBase)
				{
					CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.pass.sbt.missRecordBase)));
				}
				if (state.pass.sbt.hitgroupRecordBase)
				{
					CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.pass.sbt.hitgroupRecordBase)));
				}

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
					missRecordsList.push_back(occlusionRecord);

					MissSbtRecord radianceRecord = {};
					OPTIX_CHECK(optixSbtRecordPackHeader(state.pass.missRadianceProgGroup, &radianceRecord));
					missRecordsList.push_back(radianceRecord);

					MissSbtRecord shadowRecord = {};
					OPTIX_CHECK(optixSbtRecordPackHeader(state.pass.missShadowProgGroup, &shadowRecord));
					missRecordsList.push_back(shadowRecord);
				}

				List<HitGroupSbtRecord> hitgroupRecordsList = {};
				for (auto& instanceData : state.instanceDatas)
				{
					auto& data = state.meshDatas[instanceData.mesh->GetObjectId()];
					uint32_t submeshCount = instanceData.renderer->GetMesh()->GetSubMeshCount();
					for (uint32_t i = 0; i < submeshCount; ++i)
					{
						Material* material = instanceData.renderer->GetMaterial(i); // if empty or no base map need white texture
						Texture* texture = material->GetTexture(s_BaseMapId);

						HitGroupSbtRecord occlusionRecord = {};
						occlusionRecord.data.vertices = (float3*)data.vertexBuffer.data;
						occlusionRecord.data.normals = (float3*)data.normalBuffer.data;
						occlusionRecord.data.tangents = (float4*)data.tangentBuffer.data;
						occlusionRecord.data.uvs = (float2*)data.uvBuffer.data;
						occlusionRecord.data.indices = (uint3*)data.indexBuffer.data;
						OPTIX_CHECK(optixSbtRecordPackHeader(state.pass.hitgroupOcclusionProgGroup, &occlusionRecord));
						hitgroupRecordsList.push_back(occlusionRecord);

						HitGroupSbtRecord radianceRecord = {};
						radianceRecord.data.vertices = (float3*)data.vertexBuffer.data;
						radianceRecord.data.normals = (float3*)data.normalBuffer.data;
						radianceRecord.data.tangents = (float4*)data.tangentBuffer.data;
						radianceRecord.data.uvs = (float2*)data.uvBuffer.data;
						radianceRecord.data.indices = (uint3*)data.indexBuffer.data;
						radianceRecord.data.materialColor = texture == nullptr ? nullptr : (uchar4*)state.textureDatas[texture->GetObjectId()].buffer.data;
						OPTIX_CHECK(optixSbtRecordPackHeader(state.pass.hitgroupRadianceProgGroup, &radianceRecord));
						hitgroupRecordsList.push_back(radianceRecord);

						HitGroupSbtRecord shadowRecord = {};
						shadowRecord.data.vertices = (float3*)data.vertexBuffer.data;
						shadowRecord.data.normals = (float3*)data.normalBuffer.data;
						shadowRecord.data.tangents = (float4*)data.tangentBuffer.data;
						shadowRecord.data.uvs = (float2*)data.uvBuffer.data;
						shadowRecord.data.indices = (uint3*)data.indexBuffer.data;
						OPTIX_CHECK(optixSbtRecordPackHeader(state.pass.hitgroupShadowProgGroup, &shadowRecord));
						hitgroupRecordsList.push_back(shadowRecord);
					}
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
				state.pass.sbt.missRecordCount = static_cast<uint32_t>(missRecordsList.size());
				state.pass.sbt.missRecordStrideInBytes = static_cast<uint32_t>(missRecordSize);
				state.pass.sbt.hitgroupRecordBase = hitgroupRecords;
				state.pass.sbt.hitgroupRecordCount = static_cast<uint32_t>(hitgroupRecordsList.size());
				state.pass.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroupRecordSize);
			}
		}
	}

	void InitializeLights(LightmapperState& state, Scene* scene)
	{
		DirectionalLight directionalLight = {};
		List<PointLight> pointLights = {};
		for (Light* light : scene->GetComponents<Light>())
		{
			Transform* transform = light->GetTransform();
			LightType type = light->GetType();
			Color color = light->GetColor().ToLinear();
			float intensity = light->GetIntensity();

			if (type == LightType::Point)
			{
				Vector3 pos = transform->GetPosition();
				PointLight pointLight = {};
				pointLight.position = make_float3(pos.x, pos.y, pos.z);
				pointLight.color = make_float3(color.R() * intensity, color.G() * intensity, color.B() * intensity);
				pointLights.push_back(std::move(pointLight));
			}
			else if (type == LightType::Directional)
			{
				Vector3 dir = Vector3::Transform(Vector3::Backward, transform->GetRotation());
				dir.Normalize();
				directionalLight.direction = make_float3(dir.x, dir.y, dir.z);
				directionalLight.color = make_float3(color.R() * intensity, color.G() * intensity, color.B() * intensity);
			}
		}
		state.lightmappingParams.directionalLight = directionalLight;
		size_t pointLightCount = pointLights.size();
		if (pointLightCount > 0)
		{
			state.pointLights.resize_and_upload(pointLights.data(), pointLightCount);
			state.lightmappingParams.pointLights = reinterpret_cast<PointLight*>(state.pointLights.data);
		}
		state.lightmappingParams.pointLightCount = static_cast<unsigned int>(pointLightCount);

		auto it = state.textureDatas.find(INVALID_ID);
		if (state.hasSky && it != state.textureDatas.end())
		{
			state.lightmappingParams.skyColor = (uchar4*)it->second.buffer.data;
		}
		else
		{
			state.lightmappingParams.skyColor = nullptr;
		}
	}

	void InitializeLightmapBuffers(LightmapperState& state)
	{
		state.lightmappingParams.lightmapSize.x = state.params.lightmapSize;
		state.lightmappingParams.lightmapSize.y = state.params.lightmapSize;
		state.lightmappingParams.texelPerUnit = state.params.texelPerUnit;
		state.denoisingParams.imageSize.x = state.params.lightmapSize;
		state.denoisingParams.imageSize.y = state.params.lightmapSize;

		size_t lightmapColorSize = state.lightmappingParams.lightmapSize.x * state.lightmappingParams.lightmapSize.y * sizeof(float4);
		size_t lightmapNormalSize = state.lightmappingParams.lightmapSize.x * state.lightmappingParams.lightmapSize.y * sizeof(float3);

		state.colorBuffer.resize_if_needed(lightmapColorSize);
		state.normalBuffer.resize_if_needed(lightmapNormalSize);
		state.positionBuffer.resize_if_needed(lightmapColorSize);
		state.denoisedColorBuffer.resize_if_needed(lightmapColorSize);
		state.chartIndexBuffer.resize_if_needed(lightmapColorSize / 4);
		state.lightmappingParams.color = reinterpret_cast<float4*>(state.colorBuffer.data);
		state.lightmappingParams.normal = reinterpret_cast<float3*>(state.normalBuffer.data);
		state.lightmappingParams.position = reinterpret_cast<float4*>(state.positionBuffer.data);
		state.denoisingParams.inputColor = reinterpret_cast<float4*>(state.colorBuffer.data);
		state.denoisingParams.inputNormal = reinterpret_cast<float3*>(state.normalBuffer.data);
		state.denoisingParams.inputPosition = reinterpret_cast<float4*>(state.positionBuffer.data);
		state.denoisingParams.outputColor = reinterpret_cast<float4*>(state.denoisedColorBuffer.data);
		state.denoisingParams.chartIndex = reinterpret_cast<unsigned int*>(state.chartIndexBuffer.data);
	}

	bool CompareCharts(LightmappingChartData& c1, LightmappingChartData& c2)
	{
		return c1.size.x + c1.size.y > c2.size.x + c2.size.y;
	}

	bool IsChartFits(int i, int j, LightmappingChartData& chartData, List<uint64_t>& atlasMask, int blocksPerRow)
	{
		int startBlock = i / 64;
		int bitOffset = i % 64;
		Vector2Int maskSize = chartData.maskSize;

		if (bitOffset == 0)
		{
			for (int y = 0; y < maskSize.y; y++)
			{
				for (int x = 0; x < maskSize.x; ++x)
				{
					uint64_t chartValue = chartData.mask[y * maskSize.x + x];
					uint64_t atlasValue = atlasMask[(j + y) * blocksPerRow + startBlock + x];
					if (atlasValue & chartValue)
					{
						return false;
					}
				}
			}
		}
		else
		{
			for (int y = 0; y < maskSize.y; y++)
			{
				for (int x = 0; x < maskSize.x + 1; ++x)
				{
					/*int atlasBlockIndex = (j + y) * blocksPerRow + startBlock + x;
					uint64_t chartValue = chartData.mask[y * maskSize.x + x];
					uint64_t atlasValue = atlasMask[atlasBlockIndex] >> bitOffset;
					if ((startBlock + x) < blocksPerRow - 1)
					{
						atlasValue |= atlasMask[atlasBlockIndex + 1] >> (64 - bitOffset);
					}*/

					int atlasBlockIndex = (j + y) * blocksPerRow + startBlock + x;
					uint64_t chartValue = 0;
					if (x < maskSize.x)
					{
						chartValue = chartData.mask[y * maskSize.x + x] << bitOffset;
					}
					if (x > 0)
					{
						chartValue |= chartData.mask[y * maskSize.x + x - 1] >> (64 - bitOffset);
					}
					uint64_t atlasValue = atlasMask[atlasBlockIndex];
					if (atlasValue & chartValue)
					{
						return false;
					}
				}
			}
		}
		return true;
	}

	void WriteChart(int i, int j, LightmappingChartData& chartData, List<uint64_t>& atlasMask, List<uint32_t>& chartIndexMask, List<uint2>& validTexels, int blocksPerRow, uint32_t maskChartOffset)
	{
		int size = blocksPerRow * 64;
		int startBlock = i / 64;
		int bitOffset = i % 64;
		Vector2Int chartSize = chartData.size;
		Vector2Int maskSize = chartData.maskSize;

		if (bitOffset == 0)
		{
			for (int y = 0; y < maskSize.y; y++)
			{
				for (int x = 0; x < maskSize.x; ++x)
				{
					uint64_t chartValue = chartData.mask[y * maskSize.x + x];
					atlasMask[(j + y) * blocksPerRow + startBlock + x] |= chartValue;

					for (int z = 0; z < 64; ++z)
					{
						if (chartValue & (1ull << z))
						{
							int32_t index = (j + y) * size + (i + (x * 64 + z));
							chartIndexMask[index] = maskChartOffset;
							validTexels.push_back(make_uint2(i + (x * 64 + z), j + y));
						}
					}
				}
			}
		}
		else
		{
			for (int y = 0; y < maskSize.y; y++)
			{
				for (int x = 0; x < maskSize.x + 1; ++x)
				{
					uint64_t chartValue = 0;
					if (x < maskSize.x)
					{
						chartValue = chartData.mask[y * maskSize.x + x] << bitOffset;
					}
					if (x > 0)
					{
						chartValue |= chartData.mask[y * maskSize.x + (x - 1)] >> (64 - bitOffset);
					}
					atlasMask[(j + y) * blocksPerRow + startBlock + x] |= chartValue;

					for (int z = 0; z < 64; ++z)
					{
						if (chartValue & (1ull << z))
						{
							int32_t index = (j + y) * size + (i + (x * 64 + z - bitOffset));
							chartIndexMask[index] = maskChartOffset;
							validTexels.push_back(make_uint2(i + (x * 64 + z - bitOffset), j + y));
						}
					}
				}
			}
		}
	}

	void InitializeLightmapChartsAndBVH(LightmapperState& state)
	{
		List<LightmappingChartData> atlasCharts = {};

		float texelPerUnit = state.params.texelPerUnit;
		for (auto& instanceData : state.instanceDatas)
		{
			if (!instanceData.renderer->IsBakeable() || !instanceData.mesh->GetLayout().Has(VertexAttribute::Texcoord1))
			{
				continue;
			}

			Mesh* mesh = instanceData.mesh;
			auto& meshData = state.meshDatas[mesh->GetObjectId()];
			Matrix localToWorld = instanceData.transform->GetLocalToWorldMatrix();

			Vector3* vertices = mesh->GetVertices();
			uint32_t indexCount = mesh->GetIndexCount();
			uint32_t* indices = mesh->GetIndices();
			Vector3* uvs = reinterpret_cast<Vector3*>(mesh->GetUVs(1));
			if (uvs == nullptr)
			{
				continue;
			}

			state.result.chartInstanceOffset.insert_or_assign(instanceData.renderer->GetObjectId(), static_cast<uint32_t>(atlasCharts.size() + 1));
			for (uint32_t i = 0; i < meshData.chartBounds.size(); ++i)
			{
				LightmappingChartData chartData = {};
				Vector4 meshChartBounds = meshData.chartBounds[i];
				if (meshChartBounds == Vector4::Zero)
				{
					chartData.index = static_cast<uint32_t>(atlasCharts.size() + 1);
					atlasCharts.push_back(std::move(chartData));
					continue;
				}
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

				Vector2 chartCenter = (Vector2(meshChartBounds.x, meshChartBounds.y) + Vector2(meshChartBounds.z, meshChartBounds.w)) / 2;
				List<Vector2> atlasVertices = {};
				Vector4 atlasChartBounds = { FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX };

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
						atlasVertices.push_back(vertex);
					}
				}

				const float epsilon = 0.5f + s_Padding;
				Vector4Int roundedAtlasChartBounds = Vector4Int(static_cast<int>(std::floorf(atlasChartBounds.x - epsilon)), static_cast<int>(std::floorf(atlasChartBounds.y - epsilon)), static_cast<int>(std::ceilf(atlasChartBounds.z + epsilon)), static_cast<int>(std::ceilf(atlasChartBounds.w + epsilon)));
				Vector2Int roundedSize = Vector2Int(roundedAtlasChartBounds.z - roundedAtlasChartBounds.x, roundedAtlasChartBounds.w - roundedAtlasChartBounds.y);
				roundedSize.x = Math::NextDivisableBy(roundedSize.x, 4);
				roundedSize.y = Math::NextDivisableBy(roundedSize.y, 4);
				roundedAtlasChartBounds = Vector4Int(-roundedSize.x / 2, -roundedSize.y / 2, roundedSize.x / 2, roundedSize.y / 2);

				chartData.meshBounds = meshChartBounds;
				chartData.maskBounds = atlasChartBounds;
				chartData.size = Vector2Int(roundedAtlasChartBounds.z - roundedAtlasChartBounds.x, roundedAtlasChartBounds.w - roundedAtlasChartBounds.y);
				chartData.maskSize = Vector2Int(chartData.size.x / 64 + (chartData.size.x % 64 > 0 ? 1 : 0), chartData.size.y);
				chartData.mask.resize(chartData.maskSize.x * chartData.maskSize.y);
				chartData.index = static_cast<uint32_t>(atlasCharts.size() + 1);
				Vector2Int halfSize = Vector2Int(chartData.size.x / 2, chartData.size.y / 2);
				uint32_t chartVertexCount = static_cast<uint32_t>(atlasVertices.size());

				for (int x = 0; x < chartData.size.x; x += 4)
				{
					for (int y = 0; y < chartData.size.y; y += 4)
					{
						Vector4 texelBounds = Vector4(static_cast<float>(x - halfSize.x), static_cast<float>(y - halfSize.y), static_cast<float>(x - halfSize.x + 4), static_cast<float>(y - halfSize.y + 4));//Vector4(x - halfSize.x - 0.5f - s_Padding, y - halfSize.y - 0.5f - s_Padding, x - halfSize.x + 1.5f + s_Padding, y - halfSize.y + 1.5f + s_Padding);
						bool intersects = false;
						for (uint32_t j = 0; j < chartVertexCount; j += 3)
						{
							if (MathHelper::Intersects(texelBounds, atlasVertices[j], atlasVertices[j + 1], atlasVertices[j + 2]))
							{
								intersects = true;
								break;
							}
						}
						if (intersects)
						{
							int block = x / 64;
							int bit = x % 64;
							for (int y1 = 0; y1 < 4; ++y1)
							{
								chartData.mask[(y + y1) * chartData.maskSize.x + block] |= 15ull << bit;
							}
						}
					}
				}
				atlasCharts.push_back(std::move(chartData));
			}
		}

		uint32_t offset = 0;
		uint32_t chartOffset = 0;
		List<Vector3> uvs = {};

		int32_t size = state.params.lightmapSize;
		int32_t blocksPerRow = size / 64;
		std::sort(atlasCharts.begin(), atlasCharts.end(), CompareCharts);
		state.atlasMask.resize(size * blocksPerRow);
		state.chartIndexMask.resize(size * size);

		std::fill(state.atlasMask.begin(), state.atlasMask.end(), 0);
		std::fill(state.chartIndexMask.begin(), state.chartIndexMask.end(), 0);
		state.validTexels.clear();

		uint32_t maskChartOffset = 1;
		uint32_t row = 0;
		uint32_t lastChartSize = 0;

		state.result.chartOffsetScale.resize(atlasCharts.size() + 1);
		for (auto& chart : atlasCharts)
		{
			if (chart.size.x > 0)
			{
				// Reset to top if chart is smaller
				if (lastChartSize > static_cast<uint32_t>(chart.size.x + chart.size.y))
				{
					row = 0;
				}
				bool placeFound = false;
				uint32_t rows = size - chart.size.y;
				for (uint32_t j = row; j < rows; j += 4)
				{
					uint32_t columns = size - chart.size.x;
					for (uint32_t i = 0; i < columns; i += 4)
					{
						if (state.chartIndexMask[j * size + i] == 0)
						{
							// Check chart
							if (IsChartFits(i, j, chart, state.atlasMask, blocksPerRow))
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
								WriteChart(i, j, chart, state.atlasMask, state.chartIndexMask, state.validTexels, blocksPerRow, maskChartOffset);
								break;
							}
						}
					}
					if (placeFound)
					{
						break;
					}
				}
			}
			else
			{
				state.result.chartOffsetScale[chart.index] = Vector4::Zero;
			}
			lastChartSize = chart.size.x + chart.size.y;
			++maskChartOffset;
		}

		uint32_t instanceOffset = 0;
		maskChartOffset = 1;
		for (auto& instanceData : state.instanceDatas)
		{
			if (!instanceData.renderer->IsBakeable() || !instanceData.mesh->GetLayout().Has(VertexAttribute::Texcoord1))
			{
				instanceOffset += 1;
				continue;
			}

			Mesh* mesh = instanceData.mesh;
			auto& meshData = state.meshDatas[mesh->GetObjectId()];
			uint32_t vertexCount = mesh->GetVertexCount();
			Vector3* ligthmapUvs = reinterpret_cast<Vector3*>(mesh->GetUVs(1));
			if (ligthmapUvs == nullptr)
			{
				instanceOffset += 1;
				continue;
			}

			for (uint32_t i = 0; i < vertexCount; ++i)
			{
				Vector3 uv = ligthmapUvs[i];
				Vector4 scaleOffset = state.result.chartOffsetScale[maskChartOffset + static_cast<uint32_t>(uv.z)];
				uv.x = uv.x * scaleOffset.z + scaleOffset.x;
				uv.y = uv.y * scaleOffset.w + scaleOffset.y;
				uvs.push_back(uv);
			}

			CUDABVHInput bvhInput = { uvs.data(), mesh->GetIndices(), vertexCount, mesh->GetIndexCount(), meshData.vertexBuffer.data, meshData.normalBuffer.data, meshData.tangentBuffer.data, meshData.indexBuffer.data, instanceOffset };
			state.bvh.AddInstance(bvhInput);
			uvs.clear();
			instanceOffset += 1;
			maskChartOffset += static_cast<uint32_t>(meshData.chartBounds.size());
		}

		state.bvh.Build();
		state.lightmappingParams.bvh = state.bvh.bvh;
		state.validTexelsBuffer.resize(state.validTexels.size() * sizeof(uint2));
		state.validTexelsBuffer.upload(state.validTexels.data(), state.validTexels.size());
		state.lightmappingParams.validTexels = reinterpret_cast<uint2*>(state.validTexelsBuffer.data);
		state.lightmappingParams.validTexelsCount = static_cast<uint32_t>(state.validTexels.size());
		state.chartIndexBuffer.upload(state.chartIndexMask.data(), state.chartIndexMask.size());
	}

	void InitializeProbes(LightmapperState& state, Scene* scene, const List<Vector3>& positions)
	{
		state.lightmappingParams.probeVolumeSize = make_uint3(state.params.probeVolumeSize.x, state.params.probeVolumeSize.y, state.params.probeVolumeSize.z);
		state.probePositionBuffer.resize_and_upload(positions.data(), positions.size());
		state.probeColorBuffer.resize(positions.size() * 6 * sizeof(uint32_t));

		state.lightmappingParams.probePosition = reinterpret_cast<float3*>(state.probePositionBuffer.data);
		state.lightmappingParams.probeColor = reinterpret_cast<unsigned int*>(state.probeColorBuffer.data);
		state.lightmappingParams.probeCount = static_cast<uint32_t>(positions.size());
	}

	void Launch(LightmapperState& state)
	{
		state.lightmappingParams.samplePerTexel = state.params.samplePerTexel;

		// Launch
		{
			cudaHostAlloc(&s_State.completeTexelCounter, sizeof(unsigned int), cudaHostAllocMapped | cudaHostAllocPortable);
			cudaHostAlloc(&s_State.completeProbeCounter, sizeof(unsigned int), cudaHostAllocMapped | cudaHostAllocPortable);
			*s_State.completeTexelCounter = 0;
			*s_State.completeProbeCounter = 0;
			cudaHostGetDevicePointer(&s_State.lightmappingParams.completeTexelCounter, s_State.completeTexelCounter, 0);
			cudaHostGetDevicePointer(&s_State.lightmappingParams.completeProbeCounter, s_State.completeProbeCounter, 0);
			
			if (s_ParamsPtr == 0)
			{
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_ParamsPtr), sizeof(LightmappingParams)));
			}

			CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void*>(s_ParamsPtr),
				&state.lightmappingParams, sizeof(LightmappingParams),
				cudaMemcpyHostToDevice
			));

			uint32_t launchSize = static_cast<uint32_t>(state.params.tileSize * state.params.tileSize);
			OPTIX_CHECK(optixLaunch(state.pass.pipeline, state.stream, s_ParamsPtr, sizeof(LightmappingParams), &state.pass.sbt, launchSize, 1, 1));
			cudaError_t err = cudaGetLastError();
			CUDA_SYNC_CHECK();
			
			cudaFreeHost(s_State.completeTexelCounter);
			cudaFreeHost(s_State.completeProbeCounter);
			s_State.completeTexelCounter = nullptr;
			s_State.completeProbeCounter = nullptr;
			s_State.lightmappingParams.completeTexelCounter = nullptr;
			s_State.lightmappingParams.completeProbeCounter = nullptr;
		}
	}

	//void LaunchDenoiser(LightmapperState& state)
	//{
	//	// Launch
	//	{
	//		if (s_DenoisingParamsPtr == 0)
	//		{
	//			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_DenoisingParamsPtr), sizeof(DenoisingParams)));
	//		}

	//		size_t size;
	//		CUDA_CHECK_RESULT(cuModuleGetGlobal(&s_DenoisingParamsPtr, &size, state.denoiserPtxModule, "params"));

	//		CUDA_CHECK_RESULT(cuMemcpyHtoD(s_DenoisingParamsPtr, &state.denoisingParams, sizeof(DenoisingParams)));
	//		CUDA_CHECK_RESULT(cuLaunchKernel(state.denoiserKernelFirstPass, state.lightmappingParams.imageSize.x / 8, state.lightmappingParams.imageSize.y / 8, 1, 8, 8, 1, 0, 0, 0, 0));
	//		CUDA_CHECK_RESULT(cuCtxSynchronize());

	//		float4* input = state.denoisingParams.inputColor;
	//		state.denoisingParams.inputColor = state.denoisingParams.outputColor;
	//		state.denoisingParams.outputColor = input;

	//		CUDA_CHECK_RESULT(cuMemcpyHtoD(s_DenoisingParamsPtr, &state.denoisingParams, sizeof(DenoisingParams)));
	//		CUDA_CHECK_RESULT(cuLaunchKernel(state.denoiserKernelpass, state.lightmappingParams.imageSize.x / 8, state.lightmappingParams.imageSize.y / 8, 1, 8, 8, 1, 0, 0, 0, 0));
	//		CUDA_CHECK_RESULT(cuCtxSynchronize());
	//	}
	//}

	void FixCharts(Vector4* temp, Vector4* result, const Vector2Int& resultSize)
	{
		int atlasSize = static_cast<int>(s_State.chartIndexMask.size());
		for (int j = 0; j < resultSize.y; ++j)
		{
			for (int i = 0; i < resultSize.x; ++i)
			{
				int index = j * resultSize.x + i;
				Vector4 center = *(temp + index);
				uint32_t centerChartIndex = s_State.chartIndexMask[index];

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
							uint32_t nearbyChartIndex = s_State.chartIndexMask[nearbyIndex];
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
						Vector4 color = result[index];
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
					mean /= static_cast<float>(validCount);
					for (int x = 0; x < 4; ++x)
					{
						for (int y = 0; y < 4; ++y)
						{
							int index = (j + y) * resultSize.x + (i + x);
							if (result[index].w < 0.5)
							{
								result[index] = mean;
							}
						}
					}
				}
			}
		}
	}

	void PostprocessLightmap(LightmapperState& state)
	{
		int outputSize = state.result.lightmapOutputSize.x * state.result.lightmapOutputSize.y;
		List<Vector4> temp(outputSize);
		state.colorBuffer.download(temp.data(), (outputSize * sizeof(float4)) / sizeof(Vector4));
		state.result.lightmapOutput.resize(outputSize * (sizeof(Vector4) / sizeof(uint8_t)));
		FixCharts(temp.data(), reinterpret_cast<Vector4*>(state.result.lightmapOutput.data()), state.result.lightmapOutputSize);
	}

	void PostprocessProbes(LightmapperState& state)
	{
		state.result.probeOutput.resize(state.result.probeOutputSize * sizeof(uint32_t));
		state.probeColorBuffer.download(state.result.probeOutput.data(), state.result.probeOutputSize * sizeof(uint32_t));
	}

	void LightmappingManager::BakeLightmap(Scene* scene, const CalculationParams& params)
	{
		if (s_LightmappingState != LightmappingState::None)
		{
			return;
		}
		CalculationResult result = {};
		result.lightmapOutputSize = Vector2Int(params.lightmapSize, params.lightmapSize);
		result.hasLightmap = true;
		s_State.params = params;
		s_State.result = result;

		InitializeState(s_State);
		CreateTextures(s_State, scene);
		
		s_LightmappingState = LightmappingState::Calculating;
		std::thread worker([scene]()
		{
			BB_INITIALIZE_ALLOCATOR_THREAD();
			CreateAccelerationStructures(s_State, scene);
			CreateSBT(s_State);
			InitializeLights(s_State, scene);
			InitializeLightmapBuffers(s_State);
			InitializeLightmapChartsAndBVH(s_State);
			Launch(s_State);
			PostprocessLightmap(s_State);
			BB_SHUTDOWN_ALLOCATOR_THREAD();
			s_LightmappingState = LightmappingState::Waiting;
		});
		worker.detach();
	}

	void LightmappingManager::BakeProbes(Scene* scene, const CalculationParams& params, const List<Vector3>& positions)
	{
		if (s_LightmappingState != LightmappingState::None)
		{
			return;
		}
		CalculationResult result = {};
		result.probeOutputSize = positions.size() * 6;
		result.hasProbes = true;
		s_State.params = params;
		s_State.result = result;

		InitializeState(s_State);
		CreateTextures(s_State, scene);

		s_LightmappingState = LightmappingState::Calculating;
		std::thread worker([scene, positions]()
		{
			BB_INITIALIZE_ALLOCATOR_THREAD();
			CreateAccelerationStructures(s_State, scene);
			CreateSBT(s_State);
			InitializeLights(s_State, scene);
			InitializeProbes(s_State, scene, positions);
			Launch(s_State);
			PostprocessProbes(s_State);
			BB_SHUTDOWN_ALLOCATOR_THREAD();
			s_LightmappingState = LightmappingState::Waiting;
		});
		worker.detach();
	}

	void LightmappingManager::Shutdown()
	{
		s_LightmappingState = LightmappingState::None;
	}

	float LightmappingManager::GetProgress()
	{
		if (s_LightmappingState != LightmappingState::Calculating)
		{
			return 0;
		}
		if (s_State.completeTexelCounter == nullptr && s_State.completeProbeCounter == nullptr)
		{
			return 0;
		}
		if (s_State.result.hasLightmap && s_State.result.hasProbes)
		{
			float lightmapValue = static_cast<float>(*s_State.completeTexelCounter) / s_State.validTexels.size();
			float probeValue = static_cast<float>(*s_State.completeProbeCounter) / s_State.lightmappingParams.probeCount;
			return std::fminf(lightmapValue, probeValue);
		}
		else if (s_State.result.hasLightmap)
		{
			return static_cast<float>(*s_State.completeTexelCounter) / s_State.validTexels.size();
		}
		else if (s_State.result.hasProbes)
		{
			return static_cast<float>(*s_State.completeProbeCounter) / s_State.lightmappingParams.probeCount;
		}
		return 0;
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
