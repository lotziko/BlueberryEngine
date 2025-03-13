#include "bbpch.h"
#include "DefaultRenderer.h"

#include "Blueberry\Core\Screen.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\RenderTexture.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\DefaultMaterials.h"
#include "Blueberry\Graphics\RenderContext.h"
#include "Blueberry\Graphics\HBAORenderer.h"
#include "Blueberry\Graphics\ShadowAtlas.h"
#include "Blueberry\Graphics\RealtimeLights.h"
#include "Blueberry\Scene\Components\Camera.h"

#include "Blueberry\Graphics\OpenXRRenderer.h"

namespace Blueberry
{
	static RenderContext s_DefaultContext = {};
	static CullingResults s_Results = {};

	void DefaultRenderer::Initialize()
	{
		HBAORenderer::Initialize();
		s_ResolveMSAAMaterial = Material::Create(static_cast<Shader*>(AssetLoader::Load("assets/shaders/ResolveMSAA.shader")));
		s_ShadowAtlas = new ShadowAtlas(4096, 4096, 128);
	}

	void DefaultRenderer::Shutdown()
	{
		Object::Destroy(s_ResolveMSAAMaterial);
	}
	
	void DefaultRenderer::Draw(Scene* scene, Camera* camera, Rectangle viewport, Color background, RenderTexture* colorOutput, RenderTexture* depthOutput)
	{
		CameraData cameraData = {};
		cameraData.camera = camera;

		RenderTexture* colorMSAARenderTarget = nullptr;
		RenderTexture* normalMSAARenderTarget = nullptr;
		RenderTexture* depthStencilMSAARenderTarget = nullptr;

		RenderTexture* colorNormalRenderTarget = nullptr;
		RenderTexture* depthStencilRenderTarget = nullptr;
		RenderTexture* HBAORenderTarget = nullptr;

		static size_t screenColorTextureId = TO_HASH("_ScreenColorTexture");
		static size_t screenNormalTextureId = TO_HASH("_ScreenNormalTexture");
		static size_t screenDepthStencilTextureId = TO_HASH("_ScreenDepthStencilTexture");

		bool isVr = OpenXRRenderer::IsActive() && camera->m_IsVR;
		TextureDimension textureDimension = isVr ? TextureDimension::Texture2DArray : TextureDimension::Texture2D;
		uint32_t viewCount = isVr ? 2 : 1;
		Shader::SetKeyword(TO_HASH("MULTIVIEW"), isVr);

		if (isVr)
		{
			OpenXRRenderer::FillCameraData(cameraData);
			viewport = cameraData.multiviewViewport;
		}

		colorMSAARenderTarget = RenderTexture::GetTemporary(viewport.width, viewport.height, viewCount, 4, TextureFormat::R16G16B16A16_Float, textureDimension);
		normalMSAARenderTarget = RenderTexture::GetTemporary(viewport.width, viewport.height, viewCount, 4, TextureFormat::R8G8B8A8_UNorm, textureDimension);
		depthStencilMSAARenderTarget = RenderTexture::GetTemporary(viewport.width, viewport.height, viewCount, 4, TextureFormat::D24_UNorm, textureDimension);

		colorNormalRenderTarget = RenderTexture::GetTemporary(viewport.width, viewport.height, viewCount, 1, TextureFormat::R8G8B8A8_UNorm, textureDimension);
		depthStencilRenderTarget = RenderTexture::GetTemporary(viewport.width, viewport.height, viewCount, 1, TextureFormat::D24_UNorm, textureDimension);
		HBAORenderTarget = RenderTexture::GetTemporary(viewport.width, viewport.height, viewCount, 1, TextureFormat::R8G8B8A8_UNorm, textureDimension);

		s_DefaultContext.Cull(scene, cameraData, s_Results);

		// Prepare lights and shadows
		GfxDevice::SetViewCount(1);
		s_ShadowAtlas->Clear();
		RealtimeLights::PrepareShadows(s_Results, s_ShadowAtlas);

		// Draw shadows
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
		s_DefaultContext.BindCamera(s_Results, cameraData);
		s_DefaultContext.DrawRenderers(s_Results, drawingSettings);

		// Resolve depth/normal
		GfxDevice::SetRenderTarget(colorNormalRenderTarget->Get(), depthStencilRenderTarget->Get());
		GfxDevice::ClearColor({ 0.0f, 0.0f, 0.0f, 0.0f });
		GfxDevice::ClearDepth(1.0f);
		GfxDevice::SetGlobalTexture(screenNormalTextureId, normalMSAARenderTarget->Get());
		GfxDevice::SetGlobalTexture(screenDepthStencilTextureId, depthStencilMSAARenderTarget->Get());
		GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_ResolveMSAAMaterial, 0));

		// HBAO
		//HBAORenderer::Draw(depthStencilRenderTarget->Get(), colorNormalRenderTarget->Get(), camera->GetViewMatrix(), camera->GetProjectionMatrix(), viewport, HBAORenderTarget->Get());
		GfxDevice::SetRenderTarget(HBAORenderTarget->Get());
		GfxDevice::ClearColor({ 1.0f, 1.0f, 1.0f, 1.0f });
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

		if (isVr)
		{
			OpenXRRenderer::SubmitColorRenderTarget(colorNormalRenderTarget);

			float aspectRatio = static_cast<float>(viewport.height) / viewport.width;
			Rectangle eyeViewport = Rectangle(0, 0, aspectRatio * colorOutput->GetHeight(), colorOutput->GetHeight());

			GfxDevice::SetRenderTarget(colorOutput->Get());
			GfxDevice::ClearColor({ 0, 0, 0, 0 });
			GfxDevice::SetViewport(eyeViewport.x, eyeViewport.y, eyeViewport.width, eyeViewport.height);
			GfxDevice::SetGlobalTexture(screenColorTextureId, colorNormalRenderTarget->Get());
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), DefaultMaterials::GetVRMirrorView(), 0));
			GfxDevice::SetRenderTarget(nullptr);
		}
		else
		{
			if (colorOutput != nullptr)
			{
				GfxDevice::Copy(colorNormalRenderTarget->Get(), colorOutput->Get());
			}
			if (depthOutput != nullptr)
			{
				GfxDevice::Copy(depthStencilRenderTarget->Get(), depthOutput->Get());
			}
		}

		RenderTexture::ReleaseTemporary(colorMSAARenderTarget);
		RenderTexture::ReleaseTemporary(normalMSAARenderTarget);
		RenderTexture::ReleaseTemporary(depthStencilMSAARenderTarget);

		RenderTexture::ReleaseTemporary(colorNormalRenderTarget);
		RenderTexture::ReleaseTemporary(depthStencilRenderTarget);
		RenderTexture::ReleaseTemporary(HBAORenderTarget);
	}
}
