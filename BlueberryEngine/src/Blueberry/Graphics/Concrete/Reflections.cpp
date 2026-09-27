#include "Reflections.h"

#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Core\Time.h"
#include "Blueberry\Graphics\GfxTopLevelAccelerationStructure.h"
#include "Blueberry\Graphics\RayTracingShader.h"
#include "Blueberry\Graphics\ComputeShader.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\ComputeShader.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Graphics\DefaultTextures.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Scene\Components\SkyRenderer.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "PerCameraData.h"

namespace Blueberry
{
	RayTracingShader* Reflections::s_ReflectionsRayTracingShader = nullptr;
	ComputeShader* Reflections::s_ReblurComputeShaders[7] = {};
	GfxBuffer* Reflections::s_ReflectionsCameraBuffer = nullptr;
	GfxBuffer* Reflections::s_ReblurBuffer = nullptr;
	GfxTopLevelAccelerationStructure* Reflections::s_AccelerationStructure = nullptr;

	struct ReflectionsCameraData
	{
		Matrix projectionMatrix;
		Matrix inverseViewProjectionMatrix;
		Vector3 cameraPositionWS;
		unsigned int frameIndex;
		Vector2Uint viewportSize;
		Vector2Uint bufferSize;
	};

	struct ReblurSharedData
	{
		Matrix worldToClip;
		Matrix viewToClip;
		Matrix viewToWorld;
		Matrix worldToViewPrev;
		Matrix worldToClipPrev;
		Matrix worldPrevToWorld;
		Vector4 rotatorPre;
		Vector4 rotator;
		Vector4 rotatorPost;
		Vector4 frustum;
		Vector4 frustumPrev;
		Vector4 cameraDelta;
		Vector4 hitDistSettings;
		Vector4 viewVectorWorld;
		Vector4 viewVectorWorldPrev;
		Vector4 mvScale;
		Vector4 convergenceSettings;
		Vector2 antilagSettings;
		Vector2 resourceSize;
		Vector2 resourceSizeInv;
		Vector2 resourceSizeInvPrev;
		Vector2 rectSize;
		Vector2 rectSizeInv;
		Vector2 rectSizePrev;
		Vector2 resolutionScale;
		Vector2 resolutionScalePrev;
		Vector2 rectOffset;
		Vector2 jitter;
		Vector2Uint printfAt;
		Vector2Uint rectOrigin;
		Vector2Int rectSizeMinusOne;
		float disocclusionThreshold;
		float disocclusionThresholdAlternate;
		float cameraAttachedReflectionMaterialID;
		float strandMaterialID;
		float strandThickness;
		float stabilizationStrength;
		float debug;
		float orthoMode;
		float unproject;
		float denoisingRange;
		float planeDistSensitivity;
		float framerateScale;
		float minBlurRadius;
		float maxBlurRadius;
		float diffPrepassBlurRadius;
		float specPrepassBlurRadius;
		float maxAccumulatedFrameNum;
		float maxFastAccumulatedFrameNum;
		float antiFirefly;
		float lobeAngleFraction;
		float roughnessFraction;
		float historyFixFrameNum;
		float historyFixBasePixelStride;
		float historyFixAlternatePixelStride;
		float historyFixAlternatePixelStrideMaterialID;
		float fastHistoryClampingSigmaScale;
		float minRectDimMulUnproject;
		float usePrepassNotOnlyForSpecularMotionEstimation;
		float splitScreen;
		float splitScreenPrev;
		float checkerboardResolveAccumSpeed;
		float viewZScale;
		float fireflySuppressorMinRelativeScale;
		float minHitDistanceWeight;
		float diffMinMaterial;
		float specMinMaterial;
		float responsiveAccumulationInvRoughnessThreshold;
		unsigned int responsiveAccumulationMinAccumulatedFrameNum;
		unsigned int hasHistoryConfidence;
		unsigned int hasDisocclusionThresholdMix;
		unsigned int diffCheckerboard;
		unsigned int specCheckerboard;
		unsigned int frameIndex;
		unsigned int isRectChanged;
		unsigned int resetHistory;
		unsigned int returnHistoryLengthInsteadOfOcclusion;
		Vector2 dummy;
	};

	static const size_t s_ReflectionsCameraDataId = TO_HASH("ReflectionsCameraData");
	static const size_t s_BaseMapId = TO_HASH("_BaseMap");
	static const size_t s_SkyboxTextureId = TO_HASH("_SkyboxTexture");
	static const size_t s_RadianceHitDistTextureId = TO_HASH("_RadianceHitDistTexture");
	static const size_t s_NormalRoughnessTextureId = TO_HASH("_NormalRoughnessTexture");
	static const size_t s_ViewZTextureId = TO_HASH("_ViewZTexture");
	static const size_t s_ReblurInViewZId = TO_HASH("gIn_ViewZ");
	static const size_t s_ReblurOutTilesId = TO_HASH("gOut_Tiles");
	static const size_t s_ReblurClassifyTilesConstantsId = TO_HASH("REBLUR_ClassifyTilesConstants");
	static const size_t s_ReblurInTilesId = TO_HASH("gIn_Tiles");
	static const size_t s_ReblurInNormalRoughnessId = TO_HASH("gIn_Normal_Roughness");
	static const size_t s_ReblurInSpecId = TO_HASH("gIn_Spec");
	static const size_t s_ReblurOutSpecId = TO_HASH("gOut_Spec");
	static const size_t s_ReblurOutSpecHitDistForTrackingId = TO_HASH("gOut_SpecHitDistForTracking");
	static const size_t s_ReblurPrePassConstantsId = TO_HASH("REBLUR_PrePassConstants");
	static const size_t s_ReblurInMvId = TO_HASH("gIn_Mv");
	static const size_t s_ReblurInSpecHitDistForTrackingId = TO_HASH("gIn_SpecHitDistForTracking");
	static const size_t s_ReblurInDisocclusionThresholdMixId = TO_HASH("gIn_DisocclusionThresholdMix");
	static const size_t s_ReblurInSpecConfidenceId = TO_HASH("gIn_SpecConfidence");
	static const size_t s_ReblurPrevViewZId = TO_HASH("gPrev_ViewZ");
	static const size_t s_ReblurPrevNormalRoughnessId = TO_HASH("gPrev_Normal_Roughness");
	static const size_t s_ReblurPrevInternalDataId = TO_HASH("gPrev_InternalData");
	static const size_t s_ReblurHistorySpecId = TO_HASH("gHistory_Spec");
	static const size_t s_ReblurHistorySpecFastId = TO_HASH("gHistory_SpecFast");
	static const size_t s_ReblurPrevSpecHitDistForTrackingId = TO_HASH("gPrev_SpecHitDistForTracking");
	static const size_t s_ReblurOutData1Id = TO_HASH("gOut_Data1");
	static const size_t s_ReblurOutSpecFastId = TO_HASH("gOut_SpecFast");
	static const size_t s_ReblurOutData2Id = TO_HASH("gOut_Data2");
	static const size_t s_ReblurTemporalAccumulationConstantsId = TO_HASH("REBLUR_TemporalAccumulationConstants");
	static const size_t s_ReblurInData1Id = TO_HASH("gIn_Data1");
	static const size_t s_ReblurInSpecFastId = TO_HASH("gIn_SpecFast");
	static const size_t s_ReblurHistoryFixConstantsId = TO_HASH("REBLUR_HistoryFixConstants");
	static const size_t s_ReblurOutViewZId = TO_HASH("gOut_ViewZ");
	static const size_t s_ReblurBlurConstantsId = TO_HASH("REBLUR_BlurConstants");
	static const size_t s_ReblurOutNormalRoughnessId = TO_HASH("gOut_Normal_Roughness");
	static const size_t s_ReblurOutInternalDataId = TO_HASH("gOut_InternalData");
	static const size_t s_ReblurOutSpecCopyId = TO_HASH("gOut_SpecCopy");
	static const size_t s_ReblurPostBlurConstantsId = TO_HASH("REBLUR_PostBlurConstants");

	void Reflections::Initialize()
	{
		s_ReflectionsRayTracingShader = static_cast<RayTracingShader*>(AssetLoader::Load("assets/shaders/Reflections.raytrace"));
		s_ReblurComputeShaders[0] = static_cast<ComputeShader*>(AssetLoader::Load("assets/shaders/nrd/REBLUR_ClassifyTiles.compute"));
		s_ReblurComputeShaders[1] = static_cast<ComputeShader*>(AssetLoader::Load("assets/shaders/nrd/REBLUR_HitDistReconstruction.compute"));
		s_ReblurComputeShaders[2] = static_cast<ComputeShader*>(AssetLoader::Load("assets/shaders/nrd/REBLUR_PrePass.compute"));
		s_ReblurComputeShaders[3] = static_cast<ComputeShader*>(AssetLoader::Load("assets/shaders/nrd/REBLUR_TemporalAccumulation.compute"));
		s_ReblurComputeShaders[4] = static_cast<ComputeShader*>(AssetLoader::Load("assets/shaders/nrd/REBLUR_HistoryFix.compute"));
		s_ReblurComputeShaders[5] = static_cast<ComputeShader*>(AssetLoader::Load("assets/shaders/nrd/REBLUR_Blur.compute"));
		s_ReblurComputeShaders[6] = static_cast<ComputeShader*>(AssetLoader::Load("assets/shaders/nrd/REBLUR_PostBlur.compute"));

		BufferProperties reflectionsCameraBufferProperties = {};
		reflectionsCameraBufferProperties.elementCount = 1;
		reflectionsCameraBufferProperties.elementSize = sizeof(ReflectionsCameraData) * 1;
		reflectionsCameraBufferProperties.usageFlags = BufferUsageFlags::ConstantBuffer;

		GfxDevice::CreateBuffer(reflectionsCameraBufferProperties, s_ReflectionsCameraBuffer);

		BufferProperties reblurBufferProperties = {};
		reblurBufferProperties.elementCount = 1;
		reblurBufferProperties.elementSize = sizeof(ReblurSharedData) * 1;
		reblurBufferProperties.usageFlags = BufferUsageFlags::ConstantBuffer;

		GfxDevice::CreateBuffer(reblurBufferProperties, s_ReblurBuffer);
		GfxDevice::CreateTopLevelAccelerationStructure(s_AccelerationStructure);
	}

	void Reflections::Shutdown()
	{
		delete s_ReflectionsCameraBuffer;
		if (s_AccelerationStructure != nullptr)
		{
			delete s_AccelerationStructure;
		}
	}

	void Reflections::Draw(Scene* scene, Camera* camera, Rectangle viewport, Vector2Int size, PerCameraData& perCameraData)
	{
		PerCameraReflectionsData& reflectionsData = perCameraData.m_ReflectionsData;

		if (s_AccelerationStructure == nullptr)
		{
			return;
		}

		s_AccelerationStructure->Clear();
		for (auto& component : scene->GetIterator<MeshRenderer>())
		{
			MeshRenderer* meshRenderer = static_cast<MeshRenderer*>(component.second);
			s_AccelerationStructure->Add(meshRenderer->GetAccelerationStructure(), meshRenderer->GetMaterials(), meshRenderer->GetTransform()->GetLocalToWorldMatrix());
		}

		// TODO move miss into skybox material and make GfxTopLevelAccelerationStructure::Add for it
		Texture* skyboxTexture = nullptr;
		for (auto& component : scene->GetIterator<SkyRenderer>())
		{
			SkyRenderer* skyRenderer = static_cast<SkyRenderer*>(component.second);
			Material* material = skyRenderer->GetMaterial();
			if (material != nullptr)
			{
				Texture* baseMap = material->GetTexture(s_BaseMapId);
				if (baseMap != nullptr)
				{
					skyboxTexture = baseMap;
					break;
				}
			}
		}
		if (skyboxTexture == nullptr)
		{
			skyboxTexture = DefaultTextures::GetBlackCube();
		}
		GfxDevice::SetGlobalTexture(s_SkyboxTextureId, skyboxTexture->Get());

		Vector2 jitter = Vector2(0.0f, 0.0f);
		float splitScreen = 0.0f;
		Matrix projection = camera->GetProjectionMatrix();

		const Matrix cameraViewToWorld = camera->GetInverseViewMatrix();
		Matrix viewToWorld = cameraViewToWorld;
		viewToWorld._41 = 0.0f;
		viewToWorld._42 = 0.0f;
		viewToWorld._43 = 0.0f;
		Matrix worldToView = viewToWorld.Invert();

		const float invX = 1.0f / projection._11;
		const float invY = 1.0f / projection._22;

		float orthoMode;
		Vector4 frustum;
		if (camera->IsOrthographic())
		{
			orthoMode = -1.0f;
			frustum = Vector4(invX, -invY, -2.0f * invX, 2.0f * invY);
		}
		else
		{
			orthoMode = 0.0f;
			frustum = Vector4(-invX, invY, 2.0f * invX, -2.0f * invY);
		}

		unsigned int frameIndex = static_cast<unsigned int>(Time::GetFrameCount() % UINT_MAX);
		double sequence = double(frameIndex) * 0.7071067811865475;
		double blurSequence = double(frameIndex) * 0.5773502691896258;
		float blurAngle = float(blurSequence - std::floor(blurSequence)) * 1.57079632679f;

		auto MakeRotator = [](float angle)
		{
			float c = std::cos(angle);
			float s = std::sin(angle);
			return Vector4(c, s, -s, c);
		};

		bool resetHistory = false;
		if (!reflectionsData.historyValid || reflectionsData.output == nullptr || reflectionsData.previousResourceSize != size || reflectionsData.previousIsOrthographic != camera->IsOrthographic())
		{
			reflectionsData.previousViewZ.reset(GfxTexturePool::Get(size.x, size.y, 1, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R32_Float, TextureDimension::Texture2D));
			reflectionsData.previousNormalRoughness.reset(GfxTexturePool::Get(size.x, size.y, 1, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R10G10B10A2_Unorm, TextureDimension::Texture2D));
			reflectionsData.previousInternalData.reset(GfxTexturePool::Get(size.x, size.y, 1, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R16_UInt, TextureDimension::Texture2D));
			reflectionsData.specularHistory.reset(GfxTexturePool::Get(size.x, size.y, 1, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R16G16B16A16_Float, TextureDimension::Texture2D));
			reflectionsData.specularFastHistory.reset(GfxTexturePool::Get(size.x, size.y, 1, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R16_Float, TextureDimension::Texture2D));
			reflectionsData.historyStabilizedPing.reset(GfxTexturePool::Get(size.x, size.y, 1, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R16_Float, TextureDimension::Texture2D));
			reflectionsData.historyStabilizedPong.reset(GfxTexturePool::Get(size.x, size.y, 1, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R16_Float, TextureDimension::Texture2D));
			reflectionsData.specularHitdistForTrackingPing.reset(GfxTexturePool::Get(size.x, size.y, 1, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R16_Float, TextureDimension::Texture2D));
			reflectionsData.specularHitdistForTrackingPong.reset(GfxTexturePool::Get(size.x, size.y, 1, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R16_Float, TextureDimension::Texture2D));
			reflectionsData.output.reset(GfxTexturePool::Get(size.x, size.y, 1, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R16G16B16A16_Float, TextureDimension::Texture2D));
			
			reflectionsData.previousViewToWorld = cameraViewToWorld;
			reflectionsData.previousProjection = projection;
			reflectionsData.previousFrustum = frustum;
			reflectionsData.previousRectSize = Vector2(viewport.width, viewport.height);
			reflectionsData.previousResourceSize = size;
			reflectionsData.previousJitter = jitter;
			reflectionsData.previousSplitScreen = splitScreen;
			reflectionsData.previousIsOrthographic = camera->IsOrthographic();
			resetHistory = true;
		}

		Matrix previousRelativeViewToWorld = reflectionsData.previousViewToWorld;
		previousRelativeViewToWorld._41 -= cameraViewToWorld._41;
		previousRelativeViewToWorld._42 -= cameraViewToWorld._42;
		previousRelativeViewToWorld._43 -= cameraViewToWorld._43;
		Matrix previousRelativeWorldToView = previousRelativeViewToWorld.Invert();

		ReflectionsCameraData cameraData = {};
		cameraData.projectionMatrix = GfxDevice::GetGPUMatrix(camera->GetProjectionMatrix());
		cameraData.inverseViewProjectionMatrix = GfxDevice::GetGPUMatrix(camera->GetInverseViewProjectionMatrix());
		cameraData.cameraPositionWS = camera->GetTransform()->GetPosition();
		cameraData.frameIndex = frameIndex;
		cameraData.viewportSize = Vector2Uint(viewport.width, viewport.height);
		s_ReflectionsCameraBuffer->SetData(reinterpret_cast<char*>(&cameraData), sizeof(ReflectionsCameraData));

		ReblurSharedData reblurData = {};
		reblurData.worldToClip = GfxDevice::GetGPUMatrix(worldToView * projection);
		reblurData.viewToClip = GfxDevice::GetGPUMatrix(projection);
		reblurData.viewToWorld = GfxDevice::GetGPUMatrix(viewToWorld);
		reblurData.worldToViewPrev = GfxDevice::GetGPUMatrix(previousRelativeWorldToView);
		reblurData.worldToClipPrev = GfxDevice::GetGPUMatrix(previousRelativeWorldToView * reflectionsData.previousProjection);
		reblurData.worldPrevToWorld = Matrix::Identity;
		reblurData.rotatorPre = MakeRotator(float(sequence - std::floor(sequence)) * 1.57079632679f);
		reblurData.rotator = MakeRotator(blurAngle);
		reblurData.rotatorPost = MakeRotator(blurAngle + 0.39269908170f);
		reblurData.frustum = frustum;
		reblurData.frustumPrev = reflectionsData.previousFrustum;
		reblurData.cameraDelta = Vector4(previousRelativeViewToWorld._41, previousRelativeViewToWorld._42, previousRelativeViewToWorld._43,	0.0f);
		reblurData.hitDistSettings = Vector4(3.0f, 0.1f, 20.0f, 0.0f);
		reblurData.viewVectorWorld = Vector4(-viewToWorld._31, -viewToWorld._32, -viewToWorld._33, 0.0f);
		reblurData.viewVectorWorldPrev = Vector4(-reflectionsData.previousViewToWorld._31, -reflectionsData.previousViewToWorld._32, -reflectionsData.previousViewToWorld._33, 0.0f);
		reblurData.mvScale = Vector4(1.0f, 1.0f, 1.0f, 1.0f);
		reblurData.convergenceSettings = Vector4(1.0f, 0.2f, 0.8f, 0.0f);
		reblurData.antilagSettings = Vector2(2.0f, 3.0f);
		reblurData.resourceSize = Vector2(float(size.x), float(size.y));
		reblurData.resourceSizeInv = Vector2(1.0f / size.x, 1.0f / size.y);
		reblurData.resourceSizeInvPrev = Vector2(1.0f / reflectionsData.previousResourceSize.x, 1.0f / reflectionsData.previousResourceSize.y);
		reblurData.rectSize = Vector2(viewport.width, viewport.height);
		reblurData.rectSizeInv = Vector2(1.0f / viewport.width, 1.0f / viewport.height);
		reblurData.rectSizePrev = reflectionsData.previousRectSize;
		reblurData.resolutionScale = Vector2(float(viewport.width) / size.x, float(viewport.height) / size.y);
		reblurData.resolutionScalePrev = Vector2(float(reflectionsData.previousRectSize.x) / size.x, float(reflectionsData.previousRectSize.y) / size.y);
		const float resolutionScale = (std::min)(reblurData.resolutionScale.x, reblurData.resolutionScale.y);
		reblurData.rectOffset = Vector2(0.0f, 0.0f);
		reblurData.jitter = Vector2(0.0f, 0.0f);
		reblurData.printfAt = Vector2Uint(9999, 9999);
		reblurData.rectOrigin = Vector2Uint(0, 0);
		reblurData.rectSizeMinusOne = Vector2Int(int(viewport.width) - 1, int(viewport.height) - 1);
		const float thresholdBonus = 1.0f / float(viewport.height);
		reblurData.disocclusionThreshold = 0.01f + thresholdBonus;
		reblurData.disocclusionThresholdAlternate = 0.05f + thresholdBonus;
		reblurData.cameraAttachedReflectionMaterialID = 999.0f;
		reblurData.strandMaterialID = 999.0f;
		reblurData.strandThickness = 0.00008f;
		reblurData.stabilizationStrength = resetHistory ? 0.0f : 30.0f / 31.0f;
		reblurData.debug = 0.0f;
		reblurData.orthoMode = orthoMode;
		reblurData.unproject = 2.0f / (viewport.height * std::abs(projection._22));
		reblurData.denoisingRange = 500000.0f;
		reblurData.planeDistSensitivity = 0.02f;
		const float dt = (std::max)(Time::GetDeltaTime(), 0.000001f);
		reblurData.framerateScale = std::clamp((1.0f / 60.0f) / dt, 0.25f, 4.0f);
		reblurData.minBlurRadius = 1.0f;
		reblurData.maxBlurRadius = (std::max)(reblurData.minBlurRadius,	30.0f * resolutionScale);
		const float clampBlend = std::clamp(reblurData.maxBlurRadius * 0.5f, 0.0f, 1.0f);
		reblurData.diffPrepassBlurRadius = 0.0f;
		reblurData.specPrepassBlurRadius = 50.0f * resolutionScale;
		reblurData.maxAccumulatedFrameNum = resetHistory ? 0.0f : 30.0f;
		reblurData.maxFastAccumulatedFrameNum = resetHistory ? 0.0f : 6.0f;
		reblurData.antiFirefly = 1.0f;
		reblurData.lobeAngleFraction = 0.15f * 0.15f;
		reblurData.roughnessFraction = 0.15f;
		reblurData.historyFixFrameNum = 3.0f;
		reblurData.historyFixBasePixelStride = 14.0f;
		reblurData.historyFixAlternatePixelStride = 14.0f;
		reblurData.historyFixAlternatePixelStrideMaterialID = 999.0f;
		reblurData.fastHistoryClampingSigmaScale = 3.0f + (2.0f - 3.0f) * clampBlend;
		reblurData.minRectDimMulUnproject = (std::min)(viewport.width, viewport.height) * reblurData.unproject;
		reblurData.usePrepassNotOnlyForSpecularMotionEstimation = 1.0f;
		reblurData.splitScreen = splitScreen;
		reblurData.splitScreenPrev = reflectionsData.previousSplitScreen;
		const float checkerboardFactor = reblurData.framerateScale * 15.0f;
		reblurData.checkerboardResolveAccumSpeed = checkerboardFactor / (1.0f + checkerboardFactor);
		reblurData.viewZScale = 1.0;
		reblurData.fireflySuppressorMinRelativeScale = 2.0f;
		reblurData.minHitDistanceWeight = 0.1f;
		reblurData.diffMinMaterial = 4.0f;
		reblurData.specMinMaterial = 4.0f;
		reblurData.responsiveAccumulationInvRoughnessThreshold = 1000.0f;
		reblurData.responsiveAccumulationMinAccumulatedFrameNum = 3;
		reblurData.hasHistoryConfidence = 0;
		reblurData.hasDisocclusionThresholdMix = 0;
		reblurData.diffCheckerboard = 2;
		reblurData.specCheckerboard = 0;
		reblurData.frameIndex = frameIndex;
		reblurData.isRectChanged = reflectionsData.previousRectSize.x != float(viewport.width) || reflectionsData.previousRectSize.y != float(viewport.height);
		reblurData.resetHistory = resetHistory ? 1 : 0;
		reblurData.returnHistoryLengthInsteadOfOcclusion = 0;
		s_ReblurBuffer->SetData(reinterpret_cast<char*>(&reblurData), sizeof(ReblurSharedData));
		
		GfxTexture* specular1 = GfxTexturePool::Get(size.x, size.y, 1, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R16G16B16A16_Float, TextureDimension::Texture2D);
		GfxTexture* specular2 = GfxTexturePool::Get(size.x, size.y, 1, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R16G16B16A16_Float, TextureDimension::Texture2D);
		GfxTexture* normalRoughness = GfxTexturePool::Get(size.x, size.y, 1, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R10G10B10A2_Unorm, TextureDimension::Texture2D);
		GfxTexture* viewZ = GfxTexturePool::Get(size.x, size.y, 1, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R32_Float, TextureDimension::Texture2D);
		GfxTexture* tiles = GfxTexturePool::Get((size.x + 15) / 16, (size.y + 15) / 16, 1, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R8_UNorm, TextureDimension::Texture2D);
		GfxTexture* filteredRadiance = GfxTexturePool::Get(size.x, size.y, 1, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R16G16B16A16_Float, TextureDimension::Texture2D);
		GfxTexture* trackingDistance = GfxTexturePool::Get(size.x, size.y, 1, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R16_Float, TextureDimension::Texture2D);
		GfxTexture* data1 = GfxTexturePool::Get(size.x, size.y, 1, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R16_Float, TextureDimension::Texture2D);
		GfxTexture* data2 = GfxTexturePool::Get(size.x, size.y, 1, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R32_UInt, TextureDimension::Texture2D);
		GfxTexture* specularFast = GfxTexturePool::Get(size.x, size.y, 1, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R16_Float, TextureDimension::Texture2D);

		GfxTexture* previousViewZ = reflectionsData.previousViewZ.get();
		GfxTexture* previousNormalRoughness = reflectionsData.previousNormalRoughness.get();
		GfxTexture* previousInternalData = reflectionsData.previousInternalData.get();
		GfxTexture* specularHistory = reflectionsData.specularHistory.get();
		GfxTexture* specularFastHistory = reflectionsData.specularFastHistory.get();
		GfxTexture* specularHitdistForTrackingPing = reflectionsData.specularHitdistForTrackingPing.get();
		GfxTexture* specularHitdistForTrackingPong = reflectionsData.specularHitdistForTrackingPong.get();

		GfxDevice::SetGlobalTexture(s_RadianceHitDistTextureId, specular1);
		GfxDevice::SetGlobalTexture(s_NormalRoughnessTextureId, normalRoughness);
		GfxDevice::SetGlobalTexture(s_ViewZTextureId, viewZ);
		GfxDevice::SetGlobalBuffer(s_ReflectionsCameraDataId, s_ReflectionsCameraBuffer);
		GfxDevice::DispatchRays(s_ReflectionsRayTracingShader, s_AccelerationStructure, viewport.width, viewport.height, 1);

		// Classify tiles
		GfxDevice::SetGlobalTexture(s_ReblurInViewZId, viewZ);
		GfxDevice::SetGlobalTexture(s_ReblurOutTilesId, tiles);
		GfxDevice::SetGlobalBuffer(s_ReblurClassifyTilesConstantsId, s_ReblurBuffer);
		GfxDevice::Dispatch(s_ReblurComputeShaders[0], 0, (viewport.width + 15) / 16, (viewport.height + 15) / 16, 1);

		// HitDistReconstruction

		// Pre pass
		GfxDevice::SetGlobalTexture(s_ReblurInTilesId, tiles);
		GfxDevice::SetGlobalTexture(s_ReblurInNormalRoughnessId, normalRoughness);
		GfxDevice::SetGlobalTexture(s_ReblurInViewZId, viewZ);
		GfxDevice::SetGlobalTexture(s_ReblurInSpecId, specular1);
		GfxDevice::SetGlobalTexture(s_ReblurOutSpecId, filteredRadiance);
		GfxDevice::SetGlobalTexture(s_ReblurOutSpecHitDistForTrackingId, trackingDistance);
		GfxDevice::SetGlobalBuffer(s_ReblurPrePassConstantsId, s_ReblurBuffer);
		GfxDevice::Dispatch(s_ReblurComputeShaders[2], 0, (viewport.width + 15) / 16, (viewport.height + 15) / 16, 1);

		// Temporal accumulation
		GfxDevice::SetGlobalTexture(s_ReblurInTilesId, tiles);
		GfxDevice::SetGlobalTexture(s_ReblurInNormalRoughnessId, normalRoughness);
		GfxDevice::SetGlobalTexture(s_ReblurInViewZId, viewZ);
		GfxDevice::SetGlobalTexture(s_ReblurInMvId, DefaultTextures::GetBlack2D()->Get());
		GfxDevice::SetGlobalTexture(s_ReblurInSpecId, filteredRadiance);
		GfxDevice::SetGlobalTexture(s_ReblurInSpecHitDistForTrackingId, trackingDistance);
		GfxDevice::SetGlobalTexture(s_ReblurInDisocclusionThresholdMixId, viewZ);
		GfxDevice::SetGlobalTexture(s_ReblurInSpecConfidenceId, viewZ);
		GfxDevice::SetGlobalTexture(s_ReblurPrevViewZId, previousViewZ);
		GfxDevice::SetGlobalTexture(s_ReblurPrevNormalRoughnessId, previousNormalRoughness);
		GfxDevice::SetGlobalTexture(s_ReblurPrevInternalDataId, previousInternalData);
		GfxDevice::SetGlobalTexture(s_ReblurHistorySpecId, specularHistory);
		GfxDevice::SetGlobalTexture(s_ReblurHistorySpecFastId, specularFastHistory);
		GfxDevice::SetGlobalTexture(s_ReblurPrevSpecHitDistForTrackingId, specularHitdistForTrackingPing);
		GfxDevice::SetGlobalTexture(s_ReblurOutData1Id, data1);
		GfxDevice::SetGlobalTexture(s_ReblurOutSpecId, specular2);
		GfxDevice::SetGlobalTexture(s_ReblurOutSpecFastId, specularFast);
		GfxDevice::SetGlobalTexture(s_ReblurOutSpecHitDistForTrackingId, specularHitdistForTrackingPong);
		GfxDevice::SetGlobalTexture(s_ReblurOutData2Id, data2);
		GfxDevice::SetGlobalBuffer(s_ReblurTemporalAccumulationConstantsId, s_ReblurBuffer);
		GfxDevice::Dispatch(s_ReblurComputeShaders[3], 0, (viewport.width + 7) / 8, (viewport.height + 15) / 16, 1);

		// History fix
		GfxDevice::SetGlobalTexture(s_ReblurInTilesId, tiles);
		GfxDevice::SetGlobalTexture(s_ReblurInNormalRoughnessId, normalRoughness);
		GfxDevice::SetGlobalTexture(s_ReblurInData1Id, data1);
		GfxDevice::SetGlobalTexture(s_ReblurInViewZId, viewZ);
		GfxDevice::SetGlobalTexture(s_ReblurInSpecId, specular2);
		GfxDevice::SetGlobalTexture(s_ReblurInSpecFastId, specularFast);
		GfxDevice::SetGlobalTexture(s_ReblurInSpecHitDistForTrackingId, specularHitdistForTrackingPong);
		GfxDevice::SetGlobalTexture(s_ReblurOutSpecId, specular1);
		GfxDevice::SetGlobalTexture(s_ReblurOutSpecFastId, specularFastHistory);
		GfxDevice::SetGlobalBuffer(s_ReblurHistoryFixConstantsId, s_ReblurBuffer);
		GfxDevice::Dispatch(s_ReblurComputeShaders[4], 0, (viewport.width + 7) / 8, (viewport.height + 15) / 16, 1);

		// Blur
		GfxDevice::SetGlobalTexture(s_ReblurInTilesId, tiles);
		GfxDevice::SetGlobalTexture(s_ReblurInNormalRoughnessId, normalRoughness);
		GfxDevice::SetGlobalTexture(s_ReblurInViewZId, viewZ);
		GfxDevice::SetGlobalTexture(s_ReblurInData1Id, data1);
		GfxDevice::SetGlobalTexture(s_ReblurInSpecId, specular1);
		GfxDevice::SetGlobalTexture(s_ReblurOutViewZId, previousViewZ);
		GfxDevice::SetGlobalTexture(s_ReblurOutSpecId, specular2);
		GfxDevice::SetGlobalBuffer(s_ReblurBlurConstantsId, s_ReblurBuffer);
		GfxDevice::Dispatch(s_ReblurComputeShaders[5], 0, (viewport.width + 7) / 8, (viewport.height + 15) / 16, 1);

		// Post blur
		GfxDevice::SetGlobalTexture(s_ReblurInTilesId, tiles);
		GfxDevice::SetGlobalTexture(s_ReblurInNormalRoughnessId, normalRoughness);
		GfxDevice::SetGlobalTexture(s_ReblurInData1Id, data1);
		GfxDevice::SetGlobalTexture(s_ReblurInViewZId, previousViewZ);
		GfxDevice::SetGlobalTexture(s_ReblurInSpecId, specular2);
		GfxDevice::SetGlobalTexture(s_ReblurOutNormalRoughnessId, previousNormalRoughness);
		GfxDevice::SetGlobalTexture(s_ReblurOutSpecId, specularHistory);
		GfxDevice::SetGlobalTexture(s_ReblurOutInternalDataId, previousInternalData);
		GfxDevice::SetGlobalTexture(s_ReblurOutSpecCopyId, reflectionsData.output.get());
		GfxDevice::SetGlobalBuffer(s_ReblurPostBlurConstantsId, s_ReblurBuffer);
		GfxDevice::Dispatch(s_ReblurComputeShaders[6], 0, (viewport.width + 7) / 8, (viewport.height + 15) / 16, 1);

		// Temporal stabilization

		GfxTexturePool::Release(specular1);
		GfxTexturePool::Release(specular2);
		GfxTexturePool::Release(normalRoughness);
		GfxTexturePool::Release(viewZ);
		GfxTexturePool::Release(tiles);
		GfxTexturePool::Release(filteredRadiance);
		GfxTexturePool::Release(trackingDistance);
		GfxTexturePool::Release(data1);
		GfxTexturePool::Release(data2);
		GfxTexturePool::Release(specularFast);

		reflectionsData.previousViewToWorld = cameraViewToWorld;
		reflectionsData.previousProjection = projection;
		reflectionsData.previousFrustum = frustum;
		reflectionsData.previousRectSize = Vector2(viewport.width, viewport.height);
		reflectionsData.previousResourceSize = size;
		reflectionsData.previousJitter = jitter;
		reflectionsData.previousSplitScreen = splitScreen;
		reflectionsData.previousIsOrthographic = camera->IsOrthographic();
		reflectionsData.historyValid = true;
		std::swap(reflectionsData.historyStabilizedPing, reflectionsData.historyStabilizedPong);
		std::swap(reflectionsData.specularHitdistForTrackingPing, reflectionsData.specularHitdistForTrackingPong);
	}

	GfxTexture* Reflections::GetReflectionTexture(const PerCameraData& perCameraData)
	{
		return perCameraData.m_ReflectionsData.output.get();
	}
}
