#include "bbpch.h"
#include "DefaultRenderer.h"

#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\RenderTexture.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\RenderContext.h"

namespace Blueberry
{
	void DefaultRenderer::Initialize()
	{
		s_ColorMSAARenderTarget = RenderTexture::Create(1920, 1080, 4, TextureFormat::R16G16B16A16_FLOAT);
		s_DepthStencilMSAARenderTarget = RenderTexture::Create(1920, 1080, 4, TextureFormat::D24_UNorm);
		s_ColorRenderTarget = RenderTexture::Create(1920, 1080, 1, TextureFormat::R8G8B8A8_UNorm);
		s_DepthStencilRenderTarget = RenderTexture::Create(1920, 1080, 1, TextureFormat::D24_UNorm);

		s_ResolveMSAAMaterial = Material::Create((Shader*)AssetLoader::Load("assets/ResolveMSAA.shader"));
	}

	void DefaultRenderer::Shutdown()
	{
		delete s_ColorMSAARenderTarget;
		delete s_DepthStencilMSAARenderTarget;
		delete s_ColorRenderTarget;
		delete s_DepthStencilRenderTarget;

		Object::Destroy(s_ResolveMSAAMaterial);
	}

	void DefaultRenderer::Draw(Scene* scene, Camera* camera, Rectangle viewport, Color background, RenderTexture* colorOutput, RenderTexture* depthOutput)
	{
		static size_t screenColorTextureId = TO_HASH("_ScreenColorTexture");
		static size_t screenDepthStencilTextureId = TO_HASH("_ScreenDepthStencilTexture");

		RenderContext context = {};
		CullingResults results = {};
		context.Cull(scene, camera, results);
		context.Bind(results);

		DrawingSettings drawingSettings = {};
		drawingSettings.passIndex = 1;

		// Z-prepass
		GfxDevice::SetRenderTarget(nullptr, s_DepthStencilMSAARenderTarget->Get());
		GfxDevice::SetViewport(0, 0, viewport.width, viewport.height);
		GfxDevice::ClearDepth(1.0f);
		context.Draw(results, drawingSettings);

		drawingSettings.passIndex = 0;

		// Forward pass
		GfxDevice::SetRenderTarget(s_ColorMSAARenderTarget->Get(), s_DepthStencilMSAARenderTarget->Get());
		GfxDevice::ClearColor({ 0.0f, 0.0f, 0.0f, 0.0f });
		context.Draw(results, drawingSettings);

		GfxDevice::SetRenderTarget(s_ColorRenderTarget->Get(), s_DepthStencilRenderTarget->Get());
		GfxDevice::ClearColor(background);
		GfxDevice::ClearDepth(1.0f);
		GfxDevice::SetViewport(0, 0, s_ColorRenderTarget->GetWidth(), s_ColorRenderTarget->GetHeight());
		GfxDevice::SetGlobalTexture(screenColorTextureId, s_ColorMSAARenderTarget->Get());
		GfxDevice::SetGlobalTexture(screenDepthStencilTextureId, s_DepthStencilMSAARenderTarget->Get());
		// Gamma correction is done manually together with MSAA resolve to avoid using SRGB swapchain
		GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_ResolveMSAAMaterial));
		GfxDevice::SetRenderTarget(nullptr);

		if (colorOutput != nullptr)
		{
			GfxDevice::Copy(s_ColorRenderTarget->Get(), colorOutput->Get());
		}
		if (depthOutput != nullptr)
		{
			GfxDevice::Copy(s_DepthStencilRenderTarget->Get(), depthOutput->Get());
		}
	}

	RenderTexture* DefaultRenderer::GetColorMSAA()
	{
		return s_ColorMSAARenderTarget;
	}

	RenderTexture* DefaultRenderer::GetDepthStencilMSAA()
	{
		return s_DepthStencilMSAARenderTarget;
	}

	RenderTexture* DefaultRenderer::GetColor()
	{
		return s_ColorRenderTarget;
	}

	RenderTexture* DefaultRenderer::GetDepthStencil()
	{
		return s_DepthStencilRenderTarget;
	}
}
