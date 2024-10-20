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
#include "Blueberry\Scene\Components\Camera.h"

namespace Blueberry
{
	void DefaultRenderer::Initialize()
	{
		GfxDevice::CreateHBAORenderer(s_HBAORenderer);
		s_ResolveMSAAMaterial = Material::Create((Shader*)AssetLoader::Load("assets/shaders/ResolveMSAA.shader"));
	}

	void DefaultRenderer::Shutdown()
	{
		Object::Destroy(s_ResolveMSAAMaterial);
	}

	void DefaultRenderer::Draw(Scene* scene, Camera* camera, Rectangle viewport, Color background, RenderTexture* colorOutput, RenderTexture* depthOutput)
	{
		static size_t screenColorTextureId = TO_HASH("_ScreenColorTexture");
		static size_t screenNormalTextureId = TO_HASH("_ScreenNormalTexture");
		static size_t screenDepthStencilTextureId = TO_HASH("_ScreenDepthStencilTexture");

		RenderTexture* colorMSAARenderTarget = RenderTexture::GetTemporary(viewport.width, viewport.height, 4, TextureFormat::R16G16B16A16_FLOAT);
		RenderTexture* normalMSAARenderTarget = RenderTexture::GetTemporary(viewport.width, viewport.height, 4, TextureFormat::R8G8B8A8_UNorm);
		RenderTexture* depthStencilMSAARenderTarget = RenderTexture::GetTemporary(viewport.width, viewport.height, 4, TextureFormat::D24_UNorm);

		RenderTexture* colorNormalRenderTarget = RenderTexture::GetTemporary(viewport.width, viewport.height, 1, TextureFormat::R8G8B8A8_UNorm);
		RenderTexture* depthStencilRenderTarget = RenderTexture::GetTemporary(viewport.width, viewport.height, 1, TextureFormat::D24_UNorm);
		RenderTexture* HBAORenderTarget = RenderTexture::GetTemporary(viewport.width, viewport.height, 1, TextureFormat::R8G8B8A8_UNorm);

		RenderContext context = {};
		CullingResults results = {};
		context.Cull(scene, camera, results);
		context.Bind(results);

		DrawingSettings drawingSettings = {};

		// Depth/normal prepass
		GfxDevice::SetRenderTarget(normalMSAARenderTarget->Get(), depthStencilMSAARenderTarget->Get());
		GfxDevice::SetViewport(viewport.x, viewport.y, viewport.width, viewport.height);
		GfxDevice::ClearColor({ 0.0f, 0.0f, 0.0f, 0.0f });
		GfxDevice::ClearDepth(1.0f);
		drawingSettings.passIndex = 1;
		context.Draw(results, drawingSettings);

		// Resolve depth/normal
		GfxDevice::SetRenderTarget(colorNormalRenderTarget->Get(), depthStencilRenderTarget->Get());
		GfxDevice::ClearColor({ 0.0f, 0.0f, 0.0f, 0.0f });
		GfxDevice::ClearDepth(1.0f);
		GfxDevice::SetGlobalTexture(screenNormalTextureId, normalMSAARenderTarget->Get());
		GfxDevice::SetGlobalTexture(screenDepthStencilTextureId, depthStencilMSAARenderTarget->Get());
		GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_ResolveMSAAMaterial, 0));

		// HBAO
		s_HBAORenderer->Draw(depthStencilRenderTarget->Get(), colorNormalRenderTarget->Get(), camera->GetViewMatrix(), camera->GetProjectionMatrix(), viewport, HBAORenderTarget->Get());
		GfxDevice::SetGlobalTexture(TO_HASH("_ScreenOcclusionTexture"), HBAORenderTarget->Get());

		// Forward pass
		GfxDevice::SetRenderTarget(colorMSAARenderTarget->Get(), depthStencilMSAARenderTarget->Get());
		GfxDevice::ClearColor({ 0.0f, 0.0f, 0.0f, 0.0f });
		drawingSettings.passIndex = 0;
		context.Draw(results, drawingSettings);

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
