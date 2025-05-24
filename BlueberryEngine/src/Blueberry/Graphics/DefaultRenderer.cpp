#include "DefaultRenderer.h"

#include "..\Core\Screen.h"
#include "..\Assets\AssetLoader.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\RenderTexture.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "GfxDevice.h"
#include "StandardMeshes.h"
#include "DefaultMaterials.h"
#include "RenderContext.h"
#include "HBAORenderer.h"
#include "ShadowAtlas.h"
#include "RealtimeLights.h"
#include "PostProcessing.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "..\Logging\Profiler.h"

#include "..\Graphics\OpenXRRenderer.h"

namespace Blueberry
{
	static RenderContext s_DefaultContext = {};
	static CullingResults s_Results = {};

	static size_t s_ScreenColorTextureId = TO_HASH("_ScreenColorTexture");
	static size_t s_ScreenNormalTextureId = TO_HASH("_ScreenNormalTexture");
	static size_t s_ScreenDepthStencilTextureId = TO_HASH("_ScreenDepthStencilTexture");
	static size_t s_ShadowTextureId = TO_HASH("_ShadowTexture");
	static size_t s_HBAOTextureId = TO_HASH("_ScreenOcclusionTexture");
	static size_t s_MultiviewKeywordId = TO_HASH("MULTIVIEW");
	static size_t s_ShadowsKeywordId = TO_HASH("SHADOWS");

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
	
	void DefaultRenderer::Draw(Scene* scene, Camera* camera, Rectangle viewport, Color background, RenderTexture* colorOutput, RenderTexture* depthOutput, const bool& simplified)
	{
		CameraData cameraData = {};
		cameraData.camera = camera;

		RenderTexture* colorMSAARenderTarget = nullptr;
		RenderTexture* normalMSAARenderTarget = nullptr;
		RenderTexture* depthStencilMSAARenderTarget = nullptr;

		RenderTexture* colorNormalRenderTarget = nullptr;
		RenderTexture* depthStencilRenderTarget = nullptr;
		RenderTexture* HBAORenderTarget = nullptr;
		RenderTexture* resultRenderTarget = nullptr;

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

		colorMSAARenderTarget = RenderTexture::GetTemporary(size.x, size.y, viewCount, 4, TextureFormat::R16G16B16A16_Float, textureDimension);
		normalMSAARenderTarget = RenderTexture::GetTemporary(size.x, size.y, viewCount, 4, TextureFormat::R8G8B8A8_UNorm, textureDimension);
		depthStencilMSAARenderTarget = RenderTexture::GetTemporary(size.x, size.y, viewCount, 4, TextureFormat::D24_UNorm, textureDimension);

		colorNormalRenderTarget = RenderTexture::GetTemporary(size.x, size.y, viewCount, 1, TextureFormat::R16G16B16A16_Float, textureDimension);
		depthStencilRenderTarget = RenderTexture::GetTemporary(size.x, size.y, viewCount, 1, TextureFormat::D24_UNorm, textureDimension);
		HBAORenderTarget = RenderTexture::GetTemporary(size.x, size.y, viewCount, 1, TextureFormat::R8G8B8A8_UNorm, textureDimension);
		resultRenderTarget = RenderTexture::GetTemporary(size.x, size.y, viewCount, 1, TextureFormat::R8G8B8A8_UNorm, textureDimension);

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

			// Draw shadows
			Shader::SetKeyword(s_ShadowsKeywordId, true);
			s_ShadowAtlas->Draw(s_DefaultContext, s_Results);
			GfxDevice::SetGlobalTexture(s_ShadowTextureId, s_ShadowAtlas->GetAtlasTexture()->Get());
			BB_PROFILE_END();
		}
		
		// Lights are binded after shadows finished rendering to have valid shadow matrices
		RealtimeLights::BindLights(s_Results, s_ShadowAtlas);
		
		// Depth/normal prepass
		BB_PROFILE_BEGIN("Depth/normals");
		GfxDevice::SetViewCount(viewCount);
		GfxDevice::SetRenderTarget(normalMSAARenderTarget->Get(), depthStencilMSAARenderTarget->Get());
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
		GfxDevice::SetRenderTarget(colorNormalRenderTarget->Get(), depthStencilRenderTarget->Get());
		GfxDevice::ClearColor({ 0.0f, 0.0f, 0.0f, 0.0f });
		GfxDevice::ClearDepth(1.0f);
		GfxDevice::SetGlobalTexture(s_ScreenColorTextureId, normalMSAARenderTarget->Get());
		GfxDevice::SetGlobalTexture(s_ScreenDepthStencilTextureId, depthStencilMSAARenderTarget->Get());
		GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_ResolveMSAAMaterial, 0));

		// HBAO
		HBAORenderer::Draw(depthStencilRenderTarget->Get(), colorNormalRenderTarget->Get(), camera->GetViewMatrix(), camera->GetProjectionMatrix(), viewport, HBAORenderTarget->Get());
		GfxDevice::SetGlobalTexture(s_HBAOTextureId, HBAORenderTarget->Get());
		
		// Forward pass
		BB_PROFILE_BEGIN("Forward");
		GfxDevice::SetRenderTarget(colorMSAARenderTarget->Get(), depthStencilMSAARenderTarget->Get());
		GfxDevice::ClearColor({ 0.0f, 0.0f, 0.0f, 0.0f });
		s_DefaultContext.DrawSky(scene);
		drawingSettings.passIndex = 0;
		drawingSettings.sortingMode = SortingMode::Default;
		s_DefaultContext.DrawRenderers(s_Results, drawingSettings);
		BB_PROFILE_END();

		// Resolve color
		GfxDevice::SetRenderTarget(colorNormalRenderTarget->Get());
		GfxDevice::ClearColor(background);
		GfxDevice::SetGlobalTexture(s_ScreenColorTextureId, colorMSAARenderTarget->Get());
		GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_ResolveMSAAMaterial, 1));
		
		GfxDevice::SetRenderTarget(nullptr);
		PostProcessing::Draw(colorNormalRenderTarget->Get(), viewport);

		// Tonemapping
		// Gamma correction is done manually together with MSAA resolve to avoid using SRGB swapchain
		GfxDevice::SetRenderTarget(resultRenderTarget->Get());
		GfxDevice::SetViewport(0, 0, size.x, size.y);
		GfxDevice::SetGlobalTexture(s_ScreenColorTextureId, colorNormalRenderTarget->Get());
		GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), DefaultMaterials::GetPostProcessing(), simplified ? 1 : 0));
		GfxDevice::SetRenderTarget(nullptr);

		if (isVr)
		{
			OpenXRRenderer::SubmitColorRenderTarget(resultRenderTarget);

			float aspectRatio = static_cast<float>(viewport.height) / viewport.width;
			Rectangle eyeViewport = Rectangle(0, 0, aspectRatio * colorOutput->GetHeight(), colorOutput->GetHeight());

			GfxDevice::SetRenderTarget(colorOutput->Get());
			GfxDevice::ClearColor({ 0.0f, 0.0f, 0.0f, 0.0f });
			GfxDevice::SetViewport(eyeViewport.x, eyeViewport.y, eyeViewport.width, eyeViewport.height);
			GfxDevice::SetGlobalTexture(s_ScreenColorTextureId, resultRenderTarget->Get());
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), DefaultMaterials::GetVRMirrorView(), 0));
			GfxDevice::SetRenderTarget(nullptr);
		}
		else
		{
			if (colorOutput != nullptr)
			{
				GfxDevice::Copy(resultRenderTarget->Get(), colorOutput->Get());
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
		RenderTexture::ReleaseTemporary(resultRenderTarget);
	}
}
