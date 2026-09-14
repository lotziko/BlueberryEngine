#include "Blueberry\Graphics\Concrete\DefaultRenderer.h"

#include "Blueberry\Core\Screen.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Logging\Profiler.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\Texture3D.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\GfxTexturePool.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\DefaultMaterials.h"
#include "Blueberry\Graphics\DefaultTextures.h"
#include "..\RenderContext.h"
#include "ShadowAtlas.h"
#include "CookieAtlas.h"
#include "RealtimeLights.h"
#include "PostProcessing.h"
#include "VolumetricFog.h"
#include "AmbientOcclusion.h"
#include "RayTracing.h"
#include "Blueberry\Scene\Components\Camera.h"

#include "..\OpenXRRenderer.h"

namespace Blueberry
{
	static RenderContext s_DefaultContext = {};
	static CullingResults s_Results = {};

	static size_t s_ScreenColorTextureId = TO_HASH("_ScreenColorTexture");
	static size_t s_ScreenNormalWSTextureId = TO_HASH("_ScreenNormalWSTexture");
	static size_t s_ScreenORMTextureId = TO_HASH("_ScreenORMTexture");
	static size_t s_ScreenBakedGITextureId = TO_HASH("_ScreenBakedGITexture");
	static size_t s_ScreenDepthStencilTextureId = TO_HASH("_ScreenDepthStencilTexture");
	static size_t s_ShadowTextureId = TO_HASH("_ShadowTexture");
	static size_t s_CookieTextureId = TO_HASH("_CookieTexture");
	static size_t s_HBAOTextureId = TO_HASH("_ScreenOcclusionTexture");
	static size_t s_ReflectionTextureId = TO_HASH("_ScreenReflectionTexture");
	static size_t s_VolumetricFogTextureId = TO_HASH("_VolumetricFogTexture");
	static size_t s_MultiviewKeywordId = TO_HASH("MULTIVIEW");
	static size_t s_ShadowsKeywordId = TO_HASH("SHADOWS");
	static size_t s_ReflectionsKeywordId = TO_HASH("REFLECTIONS");
	static size_t s_DepthPassId = TO_HASH("Depth");
	static size_t s_ForwardPassId = TO_HASH("Forward");
	static size_t s_DeferredPassId = TO_HASH("Deferred");

	void DefaultRenderer::Initialize()
	{
		CookieAtlas::Initialize();
		PostProcessing::Initialize();
		VolumetricFog::Initialize();
		AmbientOcclusion::Initialize();
		RealtimeLights::Initialize();
		ShadowAtlas::Initialize();
		RayTracing::Initialize();
	}

	void DefaultRenderer::Shutdown()
	{
		CookieAtlas::Shutdown();
		PostProcessing::Shutdown();
		VolumetricFog::Shutdown();
		AmbientOcclusion::Shutdown();
		RealtimeLights::Shutdown();
		ShadowAtlas::Shutdown();
		RayTracing::Shutdown();
	}
	
	void DefaultRenderer::Draw(Scene* scene, Camera* camera, Rectangle viewport, GfxTexture* colorOutput, GfxTexture* depthOutput)
	{
		CameraData cameraData = {};
		cameraData.camera = camera;

		CameraType cameraType = camera->GetCameraType();

		GfxTexture* gBuffer[4] = {};
		GfxTexture* depthStencilRenderTarget = nullptr;
		GfxTexture* HBAORenderTarget = nullptr;
		GfxTexture* reflectionRenderTarget = nullptr;
		GfxTexture* postProcessingRenderTarget = nullptr;
		GfxTexture* resultRenderTarget = nullptr;

		bool isVr = OpenXRRenderer::IsActive() && cameraType == CameraType::VR;
		TextureDimension textureDimension = isVr ? TextureDimension::Texture2DArray : TextureDimension::Texture2D;
		uint32_t viewCount = isVr ? 2 : 1;
		Vector2Int size = Vector2Int(colorOutput->GetWidth(), colorOutput->GetHeight());
		Shader::SetKeyword(s_MultiviewKeywordId, isVr);

		if (isVr)
		{
			OpenXRRenderer::FillCameraData(cameraData);
			viewport = cameraData.multiviewViewport;
			size = Vector2Int(viewport.width, viewport.height);
		}
		else
		{
			cameraData.size = Vector2Int(viewport.width, viewport.height);
			cameraData.renderTargetSize = size;
		}

		gBuffer[0] = GfxTexturePool::Get(size.x, size.y, viewCount, TextureUsageFlags::RenderTarget, 1, 1, TextureFormat::R16G16B16A16_Float, textureDimension, WrapMode::Clamp, FilterMode::Bilinear);
		gBuffer[1] = GfxTexturePool::Get(size.x, size.y, viewCount, TextureUsageFlags::RenderTarget, 1, 1, TextureFormat::R16G16_UNorm, textureDimension, WrapMode::Clamp, FilterMode::Bilinear);
		gBuffer[2] = GfxTexturePool::Get(size.x, size.y, viewCount, TextureUsageFlags::RenderTarget, 1, 1, TextureFormat::R8G8B8A8_UNorm, textureDimension, WrapMode::Clamp, FilterMode::Bilinear);
		gBuffer[3] = GfxTexturePool::Get(size.x, size.y, viewCount, TextureUsageFlags::RenderTarget, 1, 1, TextureFormat::R11G11B10_Float, textureDimension, WrapMode::Clamp, FilterMode::Bilinear);
		depthStencilRenderTarget = GfxTexturePool::Get(size.x, size.y, viewCount, TextureUsageFlags::RenderTarget, 1, 1, TextureFormat::D24_UNorm, textureDimension);
		HBAORenderTarget = GfxTexturePool::Get(size.x, size.y, viewCount, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R32_UInt, textureDimension);
		postProcessingRenderTarget = GfxTexturePool::Get(size.x, size.y, viewCount, TextureUsageFlags::RenderTarget | TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R16G16B16A16_Float, textureDimension, WrapMode::Clamp, FilterMode::Bilinear);
		resultRenderTarget = GfxTexturePool::Get(size.x, size.y, viewCount, TextureUsageFlags::RenderTarget, 1, 1, colorOutput->GetFormat(), textureDimension);

		BB_PROFILE_BEGIN("Culling");
		s_DefaultContext.Cull(scene, cameraData, s_Results);
		BB_PROFILE_END();

		Shader::SetKeyword(s_ReflectionsKeywordId, cameraType != CameraType::Reflection && cameraType != CameraType::Preview);

		if (cameraType == CameraType::Preview)
		{
			Shader::SetKeyword(s_ShadowsKeywordId, false);
			GfxDevice::SetGlobalTexture(s_CookieTextureId, DefaultTextures::GetWhite3D()->Get());
			GfxDevice::SetGlobalTexture(s_VolumetricFogTextureId, DefaultTextures::GetBlack3D()->Get());
		}
		else
		{
			BB_PROFILE_BEGIN("Shadows");
			// Prepare shadows
			GfxDevice::SetViewCount(1);
			ShadowAtlas::Clear();
			RealtimeLights::PrepareShadows(s_Results);
			CookieAtlas::PrepareCookies(s_Results);
			GfxDevice::SetGlobalTexture(s_CookieTextureId, CookieAtlas::GetAtlasTexture());

			// Draw shadows
			Shader::SetKeyword(s_ShadowsKeywordId, true);
			ShadowAtlas::Draw(s_DefaultContext, s_Results);
			GfxDevice::SetGlobalTexture(s_ShadowTextureId, ShadowAtlas::GetAtlasTexture());
			BB_PROFILE_END();
		}
		
		s_DefaultContext.BindCamera(s_Results, cameraData);

		// Lights are binded after shadows finished rendering to have valid shadow matrices
		RealtimeLights::BindLights(s_Results);

		BB_PROFILE_BEGIN("Deferred");
		GfxDevice::SetViewCount(viewCount);
		GfxDevice::SetRenderTarget(gBuffer, 4, depthStencilRenderTarget);
		GfxDevice::SetViewport(viewport.x, viewport.y, viewport.width, viewport.height);
		GfxDevice::ClearColor(Color(0.0f, 0.0f, 0.0f, 0.0f));
		GfxDevice::ClearDepth(1.0f);
		DrawingSettings drawingSettings = {};
		drawingSettings.passId = s_DeferredPassId;
		drawingSettings.sortingMode = SortingMode::FrontToBack;
		drawingSettings.useGI = cameraType != CameraType::Reflection && cameraType != CameraType::Preview;
		s_DefaultContext.DrawRenderers(s_Results, drawingSettings);
		GfxDevice::SetGlobalTexture(s_ScreenColorTextureId, gBuffer[0]);
		GfxDevice::SetGlobalTexture(s_ScreenNormalWSTextureId, gBuffer[1]);
		GfxDevice::SetGlobalTexture(s_ScreenORMTextureId, gBuffer[2]);
		GfxDevice::SetGlobalTexture(s_ScreenBakedGITextureId, gBuffer[3]);
		GfxDevice::SetGlobalTexture(s_ScreenDepthStencilTextureId, depthStencilRenderTarget);
		BB_PROFILE_END();

		if (cameraType != CameraType::Preview)
		{
			reflectionRenderTarget = GfxTexturePool::Get(size.x, size.y, 1, TextureUsageFlags::UnorderedAccess);
			RayTracing::Draw(scene, camera, reflectionRenderTarget, viewport, size);
			GfxDevice::SetGlobalTexture(s_ReflectionTextureId, reflectionRenderTarget);
			VolumetricFog::CalculateFrustum(s_Results, cameraData);
			GfxDevice::SetGlobalTexture(s_VolumetricFogTextureId, VolumetricFog::GetFrustumTexture());
		}

		// Ambient Occlusion
		if (cameraType == CameraType::Preview)
		{
			GfxDevice::SetGlobalTexture(s_HBAOTextureId, DefaultTextures::GetWhite2D()->Get());
		}
		else
		{
			AmbientOcclusion::Draw(camera, depthStencilRenderTarget, gBuffer[1], HBAORenderTarget, viewport);
			GfxDevice::SetGlobalTexture(s_HBAOTextureId, HBAORenderTarget);
		}

		// Deferred pass
		BB_PROFILE_BEGIN("Deferred");
		RealtimeLights::CalculateClusters();
		GfxDevice::SetRenderTarget(postProcessingRenderTarget);
		GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), DefaultMaterials::GetDeferred()));
		GfxDevice::SetRenderTarget(postProcessingRenderTarget, depthStencilRenderTarget);
		s_DefaultContext.DrawSky(s_Results);
		BB_PROFILE_END();

		PostProcessing::Draw(camera, postProcessingRenderTarget, resultRenderTarget, viewport, cameraType);

		GfxDevice::SetRenderTarget(resultRenderTarget);
		s_DefaultContext.DrawCanvases(s_Results);

		if (isVr)
		{
			OpenXRRenderer::SubmitColorRenderTarget(resultRenderTarget);

			float aspectRatio = static_cast<float>(viewport.height) / viewport.width;
			Rectangle eyeViewport = Rectangle(0, 0, static_cast<long>(aspectRatio * colorOutput->GetHeight()), static_cast<long>(colorOutput->GetHeight()));

			GfxDevice::SetRenderTarget(colorOutput);
			GfxDevice::ClearColor(Color(0.0f, 0.0f, 0.0f, 0.0f));
			GfxDevice::SetViewport(eyeViewport.x, eyeViewport.y, eyeViewport.width, eyeViewport.height);
			GfxDevice::SetGlobalTexture(s_ScreenColorTextureId, resultRenderTarget);
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), DefaultMaterials::GetVRMirrorView(), 0));
			GfxDevice::SetRenderTarget(nullptr);
		}
		else
		{
			if (colorOutput != nullptr)
			{
				GfxDevice::Copy(resultRenderTarget, colorOutput);
			}
			if (depthOutput != nullptr)
			{
				GfxDevice::Copy(depthStencilRenderTarget, depthOutput);
			}
		}

		GfxTexturePool::Release(gBuffer[0]);
		GfxTexturePool::Release(gBuffer[1]);
		GfxTexturePool::Release(gBuffer[2]);
		GfxTexturePool::Release(gBuffer[3]);

		GfxTexturePool::Release(depthStencilRenderTarget);
		GfxTexturePool::Release(HBAORenderTarget);
		GfxTexturePool::Release(postProcessingRenderTarget);
		GfxTexturePool::Release(resultRenderTarget);

		if (cameraType != CameraType::Preview)
		{
			GfxTexturePool::Release(reflectionRenderTarget);
		}
	}
}
