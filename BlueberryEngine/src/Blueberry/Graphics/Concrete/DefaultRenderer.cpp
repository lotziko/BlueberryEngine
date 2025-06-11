#include "DefaultRenderer.h"

#include "..\..\Core\Screen.h"
#include "..\..\Assets\AssetLoader.h"
#include "..\..\Logging\Profiler.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "..\GfxDevice.h"
#include "..\GfxTexture.h"
#include "..\GfxRenderTexturePool.h"
#include "..\StandardMeshes.h"
#include "..\DefaultMaterials.h"
#include "..\RenderContext.h"
#include "..\HBAORenderer.h"
#include "ShadowAtlas.h"
#include "CookieAtlas.h"
#include "RealtimeLights.h"
#include "PostProcessing.h"
#include "VolumetricFog.h"
#include "Blueberry\Scene\Components\Camera.h"

#include "..\OpenXRRenderer.h"

namespace Blueberry
{
	static RenderContext s_DefaultContext = {};
	static CullingResults s_Results = {};

	static size_t s_ScreenColorTextureId = TO_HASH("_ScreenColorTexture");
	static size_t s_ScreenNormalTextureId = TO_HASH("_ScreenNormalTexture");
	static size_t s_ScreenDepthStencilTextureId = TO_HASH("_ScreenDepthStencilTexture");
	static size_t s_ShadowTextureId = TO_HASH("_ShadowTexture");
	static size_t s_CookieTextureId = TO_HASH("_CookieTexture");
	static size_t s_HBAOTextureId = TO_HASH("_ScreenOcclusionTexture");
	static size_t s_MultiviewKeywordId = TO_HASH("MULTIVIEW");
	static size_t s_ShadowsKeywordId = TO_HASH("SHADOWS");

	void DefaultRenderer::Initialize()
	{
		HBAORenderer::Initialize();
		CookieAtlas::Initialize();
		PostProcessing::Initialize();
		s_ResolveMSAAMaterial = Material::Create(static_cast<Shader*>(AssetLoader::Load("assets/shaders/ResolveMSAA.shader")));
		s_ShadowAtlas = new ShadowAtlas(4096, 4096, 128);
	}

	void DefaultRenderer::Shutdown()
	{
		Object::Destroy(s_ResolveMSAAMaterial);
	}
	
	void DefaultRenderer::Draw(Scene* scene, Camera* camera, Rectangle viewport, Color background, GfxTexture* colorOutput, GfxTexture* depthOutput, const bool& simplified)
	{
		CameraData cameraData = {};
		cameraData.camera = camera;

		GfxTexture* colorMSAARenderTarget = nullptr;
		GfxTexture* normalMSAARenderTarget = nullptr;
		GfxTexture* depthStencilMSAARenderTarget = nullptr;

		GfxTexture* colorNormalRenderTarget = nullptr;
		GfxTexture* depthStencilRenderTarget = nullptr;
		GfxTexture* HBAORenderTarget = nullptr;
		GfxTexture* resultRenderTarget = nullptr;

		bool isVr = OpenXRRenderer::IsActive() && camera->m_IsVR;
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

		colorMSAARenderTarget = GfxRenderTexturePool::Get(size.x, size.y, viewCount, 4, TextureFormat::R16G16B16A16_Float, textureDimension);
		normalMSAARenderTarget = GfxRenderTexturePool::Get(size.x, size.y, viewCount, 4, TextureFormat::R8G8B8A8_UNorm, textureDimension);
		depthStencilMSAARenderTarget = GfxRenderTexturePool::Get(size.x, size.y, viewCount, 4, TextureFormat::D24_UNorm, textureDimension);

		colorNormalRenderTarget = GfxRenderTexturePool::Get(size.x, size.y, viewCount, 1, TextureFormat::R16G16B16A16_Float, textureDimension);
		depthStencilRenderTarget = GfxRenderTexturePool::Get(size.x, size.y, viewCount, 1, TextureFormat::D24_UNorm, textureDimension);
		HBAORenderTarget = GfxRenderTexturePool::Get(size.x, size.y, viewCount, 1, TextureFormat::R8G8B8A8_UNorm, textureDimension);
		resultRenderTarget = GfxRenderTexturePool::Get(size.x, size.y, viewCount, 1, TextureFormat::R8G8B8A8_UNorm, textureDimension);

		BB_PROFILE_BEGIN("Culling");
		s_DefaultContext.Cull(scene, cameraData, s_Results);
		BB_PROFILE_END();

		if (simplified)
		{
			Shader::SetKeyword(s_ShadowsKeywordId, false);
		}
		else
		{
			BB_PROFILE_BEGIN("Shadows");
			// Prepare shadows
			GfxDevice::SetViewCount(1);
			s_ShadowAtlas->Clear();
			RealtimeLights::PrepareShadows(s_Results, s_ShadowAtlas);
			CookieAtlas::PrepareCookies(s_Results);
			GfxDevice::SetGlobalTexture(s_CookieTextureId, CookieAtlas::GetAtlasTexture());

			// Draw shadows
			Shader::SetKeyword(s_ShadowsKeywordId, true);
			s_ShadowAtlas->Draw(s_DefaultContext, s_Results);
			GfxDevice::SetGlobalTexture(s_ShadowTextureId, s_ShadowAtlas->GetAtlasTexture());
			BB_PROFILE_END();

			VolumetricFog::CalculateFrustum(s_Results, cameraData, s_ShadowAtlas);
		}
		
		// Lights are binded after shadows finished rendering to have valid shadow matrices
		RealtimeLights::BindLights(s_Results, s_ShadowAtlas);
		
		// Depth/normal prepass
		BB_PROFILE_BEGIN("Depth/normals");
		GfxDevice::SetViewCount(viewCount);
		GfxDevice::SetRenderTarget(nullptr, depthStencilMSAARenderTarget);
		GfxDevice::SetViewport(viewport.x, viewport.y, viewport.width, viewport.height);
		GfxDevice::ClearColor({ 0.0f, 0.0f, 0.0f, 0.0f });
		GfxDevice::ClearDepth(1.0f);
		DrawingSettings drawingSettings = {};
		drawingSettings.passIndex = 1;
		drawingSettings.sortingMode = SortingMode::FrontToBack;
		s_DefaultContext.BindCamera(s_Results, cameraData);
		s_DefaultContext.DrawRenderers(s_Results, drawingSettings);
		BB_PROFILE_END();

		// Resolve depth/normal
		GfxDevice::SetRenderTarget(colorNormalRenderTarget, depthStencilRenderTarget);
		GfxDevice::ClearColor({ 0.0f, 0.0f, 0.0f, 0.0f });
		GfxDevice::ClearDepth(1.0f);
		GfxDevice::SetGlobalTexture(s_ScreenColorTextureId, normalMSAARenderTarget);
		GfxDevice::SetGlobalTexture(s_ScreenDepthStencilTextureId, depthStencilMSAARenderTarget);
		GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_ResolveMSAAMaterial, 0));

		// HBAO
		HBAORenderer::Draw(depthStencilRenderTarget, colorNormalRenderTarget, camera->GetViewMatrix(), camera->GetProjectionMatrix(), viewport, HBAORenderTarget);
		GfxDevice::SetGlobalTexture(s_HBAOTextureId, HBAORenderTarget);

		// Forward pass
		BB_PROFILE_BEGIN("Forward");
		GfxDevice::SetRenderTarget(colorMSAARenderTarget, depthStencilMSAARenderTarget);
		GfxDevice::ClearColor({ 0.0f, 0.0f, 0.0f, 0.0f });
		s_DefaultContext.DrawSky(s_Results);
		drawingSettings.passIndex = 0;
		drawingSettings.sortingMode = SortingMode::Default;
		s_DefaultContext.DrawRenderers(s_Results, drawingSettings);
		BB_PROFILE_END();

		// Resolve color
		GfxDevice::SetRenderTarget(colorNormalRenderTarget);
		GfxDevice::ClearColor(background);
		GfxDevice::SetGlobalTexture(s_ScreenColorTextureId, colorMSAARenderTarget);
		GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_ResolveMSAAMaterial, 1));
		
		GfxDevice::SetRenderTarget(nullptr);
		PostProcessing::Draw(colorNormalRenderTarget, viewport);

		// Tonemapping
		// Gamma correction is done manually together with MSAA resolve to avoid using SRGB swapchain
		GfxDevice::SetRenderTarget(resultRenderTarget);
		GfxDevice::SetViewport(0, 0, size.x, size.y);
		GfxDevice::SetGlobalTexture(s_ScreenColorTextureId, colorNormalRenderTarget);
		GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), DefaultMaterials::GetPostProcessing(), simplified ? 1 : 0));
		GfxDevice::SetRenderTarget(nullptr);

		if (isVr)
		{
			OpenXRRenderer::SubmitColorRenderTarget(resultRenderTarget);

			float aspectRatio = static_cast<float>(viewport.height) / viewport.width;
			Rectangle eyeViewport = Rectangle(0, 0, aspectRatio * colorOutput->GetHeight(), colorOutput->GetHeight());

			GfxDevice::SetRenderTarget(colorOutput);
			GfxDevice::ClearColor({ 0.0f, 0.0f, 0.0f, 0.0f });
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

		GfxRenderTexturePool::Release(colorMSAARenderTarget);
		GfxRenderTexturePool::Release(normalMSAARenderTarget);
		GfxRenderTexturePool::Release(depthStencilMSAARenderTarget);

		GfxRenderTexturePool::Release(colorNormalRenderTarget);
		GfxRenderTexturePool::Release(depthStencilRenderTarget);
		GfxRenderTexturePool::Release(HBAORenderTarget);
		GfxRenderTexturePool::Release(resultRenderTarget);
	}
}
