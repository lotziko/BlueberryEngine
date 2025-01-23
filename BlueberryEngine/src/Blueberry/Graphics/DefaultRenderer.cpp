#include "bbpch.h"
#include "DefaultRenderer.h"

#include "Blueberry\Core\Screen.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\RenderTexture.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\RenderContext.h"
#include "Blueberry\Graphics\HBAORenderer.h"
#include "Blueberry\Graphics\ShadowAtlas.h"
#include "Blueberry\Graphics\RealtimeLights.h"
#include "Blueberry\Scene\Components\Camera.h"

namespace Blueberry
{
	static RenderContext s_DefaultContext = {};
	static CullingResults s_Results = {};

	void DefaultRenderer::Initialize()
	{
		GfxDevice::CreateHBAORenderer(s_HBAORenderer);
		s_ResolveMSAAMaterial = Material::Create((Shader*)AssetLoader::Load("assets/shaders/ResolveMSAA.shader"));
		s_ShadowAtlas = new ShadowAtlas(4096, 4096, 128);
	}

	void DefaultRenderer::Shutdown()
	{
		Object::Destroy(s_ResolveMSAAMaterial);
	}
	
	void DefaultRenderer::Draw(Scene* scene, Camera* camera, Rectangle viewport, Color background, RenderTexture* colorOutput, RenderTexture* depthOutput)
	{
		int viewCount = 2;
		Shader::SetKeyword(TO_HASH("MULTIVIEW"), true);

		static size_t screenColorTextureId = TO_HASH("_ScreenColorTexture");
		static size_t screenNormalTextureId = TO_HASH("_ScreenNormalTexture");
		static size_t screenDepthStencilTextureId = TO_HASH("_ScreenDepthStencilTexture");

		RenderTexture* colorMSAARenderTarget = RenderTexture::GetTemporary(viewport.width, viewport.height, 2, 4, TextureFormat::R16G16B16A16_Float, TextureDimension::Texture2DArray);
		RenderTexture* normalMSAARenderTarget = RenderTexture::GetTemporary(viewport.width, viewport.height, 2, 4, TextureFormat::R8G8B8A8_UNorm, TextureDimension::Texture2DArray);
		RenderTexture* depthStencilMSAARenderTarget = RenderTexture::GetTemporary(viewport.width, viewport.height, 2, 4, TextureFormat::D24_UNorm, TextureDimension::Texture2DArray);

		RenderTexture* colorNormalRenderTarget = RenderTexture::GetTemporary(viewport.width, viewport.height, 2, 1, TextureFormat::R8G8B8A8_UNorm, TextureDimension::Texture2DArray);
		RenderTexture* depthStencilRenderTarget = RenderTexture::GetTemporary(viewport.width, viewport.height, 2, 1, TextureFormat::D24_UNorm, TextureDimension::Texture2DArray);
		RenderTexture* HBAORenderTarget = RenderTexture::GetTemporary(viewport.width, viewport.height, 2, 1, TextureFormat::R8G8B8A8_UNorm, TextureDimension::Texture2DArray);

		s_DefaultContext.Cull(scene, camera, s_Results);

		// Prepare lights and shadows
		s_ShadowAtlas->Clear();
		RealtimeLights::PrepareShadows(s_Results, s_ShadowAtlas);

		// Draw shadows
		GfxDevice::SetViewCount(1);
		s_ShadowAtlas->Draw(s_DefaultContext, s_Results);
		GfxDevice::SetGlobalTexture(TO_HASH("_ShadowTexture"), s_ShadowAtlas->GetAtlasTexture()->Get());
		
		// Lights are binded after shadows finished rendering to have valid shadow matrices
		RealtimeLights::BindLights(s_Results);

		// Depth/normal prepass
		GfxDevice::SetViewCount(viewCount);
		GfxDevice::SetRenderTarget(normalMSAARenderTarget->Get(), depthStencilMSAARenderTarget->Get());
		GfxDevice::SetViewport(viewport.x, viewport.y, viewport.width, viewport.height);
		GfxDevice::ClearColor({ 0.0f, 0.0f, 0.0f, 0.0f });
		GfxDevice::ClearDepth(1.0f);
		DrawingSettings drawingSettings = {};
		drawingSettings.passIndex = 1;
		s_DefaultContext.BindCamera(s_Results);
		s_DefaultContext.DrawRenderers(s_Results, drawingSettings);

		// Resolve depth/normal
		GfxDevice::SetRenderTarget(colorNormalRenderTarget->Get(), depthStencilRenderTarget->Get());
		GfxDevice::ClearColor({ 0.0f, 0.0f, 0.0f, 0.0f });
		GfxDevice::ClearDepth(1.0f);
		GfxDevice::SetGlobalTexture(screenNormalTextureId, normalMSAARenderTarget->Get());
		GfxDevice::SetGlobalTexture(screenDepthStencilTextureId, depthStencilMSAARenderTarget->Get());
		GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_ResolveMSAAMaterial, 0));

		// HBAO
		//s_HBAORenderer->Draw(depthStencilRenderTarget->Get(), colorNormalRenderTarget->Get(), camera->GetViewMatrix(), camera->GetProjectionMatrix(), viewport, HBAORenderTarget->Get());
		GfxDevice::SetGlobalTexture(TO_HASH("_ScreenOcclusionTexture"), HBAORenderTarget->Get());

		// Forward pass
		GfxDevice::SetRenderTarget(colorMSAARenderTarget->Get(), depthStencilMSAARenderTarget->Get());
		GfxDevice::ClearColor({ 0.0f, 0.0f, 0.0f, 0.0f });
		drawingSettings.passIndex = 0;
		s_DefaultContext.DrawRenderers(s_Results, drawingSettings);

		// Resolve color
		GfxDevice::SetRenderTarget(colorNormalRenderTarget->Get());
		GfxDevice::ClearColor(background);
		GfxDevice::SetGlobalTexture(screenColorTextureId, colorMSAARenderTarget->Get());
		// Gamma correction is done manually together with MSAA resolve to avoid using SRGB swapchain
		GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_ResolveMSAAMaterial, 1));
		GfxDevice::SetRenderTarget(nullptr);

		if (colorOutput != nullptr)
		{
			GfxDevice::Copy(colorNormalRenderTarget->Get(), colorOutput->Get());
		}
		if (depthOutput != nullptr)
		{
			GfxDevice::Copy(depthStencilRenderTarget->Get(), depthOutput->Get());
		}

		RenderTexture::ReleaseTemporary(colorMSAARenderTarget);
		RenderTexture::ReleaseTemporary(normalMSAARenderTarget);
		RenderTexture::ReleaseTemporary(depthStencilMSAARenderTarget);

		RenderTexture::ReleaseTemporary(colorNormalRenderTarget);
		RenderTexture::ReleaseTemporary(depthStencilRenderTarget);
		RenderTexture::ReleaseTemporary(HBAORenderTarget);
	}
}
