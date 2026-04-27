#include "PostProcessing.h"

#include "AutoExposure.h"
#include "Blueberry\Core\Time.h"

#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\ComputeShader.h"
#include "Blueberry\Graphics\GfxTexturePool.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\DefaultMaterials.h"
#include "Blueberry\Scene\Components\Camera.h"

namespace Blueberry
{
	ComputeShader* PostProcessing::s_ResolveMSAABloomShader = nullptr;
	Material* PostProcessing::s_BloomMaterial = nullptr;
	GfxBuffer* PostProcessing::s_ResolveMSAABloomData = nullptr;
	GfxBuffer* PostProcessing::s_PostProcessingData = nullptr;
	GfxBuffer* PostProcessing::s_BloomData = nullptr;
	Texture2D* PostProcessing::s_BlueNoiseLUT = nullptr;
	Texture2D* PostProcessing::s_BRDFIntegrationLUT = nullptr;

	struct PostProcessingData
	{
		Vector4 exposureTime;
	};

	struct ResolveMSAABloomData
	{
		Vector3 bloomThreshold;
		float bloomScale;
		float exposure;
		Vector3 dummy;
	};

	struct BloomData
	{
		Vector2 texelSize;
		Vector2 dummy;
	};

	static size_t s_BlueNoiseLUTId = TO_HASH("_BlueNoiseLUT");
	static size_t s_BRDFIntegrationLUTId = TO_HASH("_BRDFIntegrationLUT");
	static size_t s_PostProcessingDataId = TO_HASH("PostProcessingData");
	static size_t s_ResolveMSAABloomDataId = TO_HASH("ResolveMSAABloomData");
	static size_t s_MSAASourceTextureId = TO_HASH("_MSAASourceTexture");
	static size_t s_ColorOutputTextureId = TO_HASH("_ColorOutputTexture");
	static size_t s_BloomOutputTextureId = TO_HASH("_BloomOutputTexture");
	static size_t s_BloomDataId = TO_HASH("BloomData");
	static size_t s_SourceTextureId = TO_HASH("_SourceTexture");
	static size_t s_SourceAdditionalTextureId = TO_HASH("_SourceAdditionalTexture");
	
	static size_t s_ScreenColorTextureId = TO_HASH("_ScreenColorTexture");
	static size_t s_ScreenBloomTextureId = TO_HASH("_ScreenBloomTexture");

	void PostProcessing::Initialize()
	{
		s_ResolveMSAABloomShader = static_cast<ComputeShader*>(AssetLoader::Load("assets/shaders/ResolveMSAABloom.compute"));
		s_BloomMaterial = Material::Create(static_cast<Shader*>(AssetLoader::Load("assets/shaders/Bloom.shader")));

		BufferProperties postProcessingBufferProperties = {};
		postProcessingBufferProperties.elementCount = 1;
		postProcessingBufferProperties.elementSize = sizeof(PostProcessingData) * 1;
		postProcessingBufferProperties.usageFlags = BufferUsageFlags::ConstantBuffer;

		GfxDevice::CreateBuffer(postProcessingBufferProperties, s_PostProcessingData);
		
		BufferProperties resolveMSAABloomBufferProperties = {};
		resolveMSAABloomBufferProperties.elementCount = 1;
		resolveMSAABloomBufferProperties.elementSize = sizeof(ResolveMSAABloomData);
		resolveMSAABloomBufferProperties.usageFlags = BufferUsageFlags::ConstantBuffer;

		GfxDevice::CreateBuffer(resolveMSAABloomBufferProperties, s_ResolveMSAABloomData);

		BufferProperties bloomBufferProperties = {};

		bloomBufferProperties.elementCount = 1;
		bloomBufferProperties.elementSize = sizeof(BloomData);
		bloomBufferProperties.usageFlags = BufferUsageFlags::ConstantBuffer;

		GfxDevice::CreateBuffer(bloomBufferProperties, s_BloomData);

		auto blueNoiseParameters = std::make_pair(false, WrapMode::Repeat); // TODO remove sampler from texture
		auto brdfParameters = std::make_pair(false, WrapMode::Clamp);
		s_BlueNoiseLUT = static_cast<Texture2D*>(AssetLoader::Load("assets/textures/BlueNoiseLUT.png", &blueNoiseParameters));
		s_BRDFIntegrationLUT = static_cast<Texture2D*>(AssetLoader::Load("assets/textures/BRDFIntegrationLUT.png", &brdfParameters));
		GfxDevice::SetGlobalTexture(s_BlueNoiseLUTId, s_BlueNoiseLUT->Get());
		GfxDevice::SetGlobalTexture(s_BRDFIntegrationLUTId, s_BRDFIntegrationLUT->Get());
		AutoExposure::Initialize();
	}

	void PostProcessing::Shutdown()
	{
		Object::Destroy(s_ResolveMSAABloomShader);
		Object::Destroy(s_BloomMaterial);
		delete s_PostProcessingData;
		delete s_ResolveMSAABloomData;
		delete s_BloomData;
		Object::Destroy(s_BlueNoiseLUT);
		Object::Destroy(s_BRDFIntegrationLUT);
		AutoExposure::Shutdown();
	}

	void PostProcessing::Draw(Camera* camera, GfxTexture* msaaColor, GfxTexture* color, GfxTexture* output, const Rectangle& viewport, const Vector2Int& size, const CameraType& cameraType)
	{
		if (cameraType == CameraType::Preview)
		{
			// Resolve color
			GfxDevice::SetRenderTarget(color);
			GfxDevice::SetGlobalTexture(s_ScreenColorTextureId, msaaColor);
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), DefaultMaterials::GetResolveMSAA(), 1));
			GfxDevice::SetRenderTarget(nullptr);

			// Tonemapping
			// Gamma correction is done manually together with MSAA resolve to avoid using SRGB swapchain
			GfxDevice::SetRenderTarget(output);
			GfxDevice::SetViewport(0, 0, size.x, size.y);
			GfxDevice::SetGlobalTexture(s_ScreenColorTextureId, color);
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), DefaultMaterials::GetPostProcessing(), 2));
			GfxDevice::SetRenderTarget(nullptr);
		}
		else
		{
			float exposure = 0.2f / AutoExposure::GetExposure(camera);
			PostProcessingData postProcessingConstants = {};
			postProcessingConstants.exposureTime = Vector4(exposure, Time::GetFrameCount() / 60.0f / 10, 0, 0);

			s_PostProcessingData->SetData(reinterpret_cast<char*>(&postProcessingConstants), sizeof(PostProcessingData));
			GfxDevice::SetGlobalBuffer(s_PostProcessingDataId, s_PostProcessingData);

			uint32_t threadWidth = viewport.width / 8 + viewport.width % 8;
			uint32_t threadHeight = viewport.height / 8 + viewport.height % 8;

			ResolveMSAABloomData resolveMSAABloomConstants = {};
			resolveMSAABloomConstants.bloomThreshold = Vector3(0, 1, 0);
			resolveMSAABloomConstants.bloomScale = 1;
			resolveMSAABloomConstants.exposure = exposure;

			s_ResolveMSAABloomData->SetData(reinterpret_cast<char*>(&resolveMSAABloomConstants), sizeof(ResolveMSAABloomData));
			GfxDevice::SetGlobalBuffer(s_ResolveMSAABloomDataId, s_ResolveMSAABloomData);

			uint32_t textureWidth = color->GetWidth() / 4;
			uint32_t textureHeight = color->GetHeight() / 4;
			uint32_t textureWidth2 = Math::NextPowerOfTwo(textureWidth);
			uint32_t textureHeight2 = Math::NextPowerOfTwo(textureHeight);

			TextureProperties textureProperties = {};

			textureProperties.width = textureWidth;
			textureProperties.height = textureHeight;
			textureProperties.depth = 1;
			textureProperties.antiAliasing = 1;
			textureProperties.mipCount = 1;
			textureProperties.format = TextureFormat::R16G16B16A16_Float;
			textureProperties.dimension = TextureDimension::Texture2D;
			textureProperties.wrapMode = WrapMode::Clamp;
			textureProperties.filterMode = FilterMode::Bilinear;
			textureProperties.usageFlags = TextureUsageFlags::RenderTarget | TextureUsageFlags::UnorderedAccess;

			GfxTexture* bloom = GfxTexturePool::Get(textureProperties);
			textureProperties.width = textureWidth2;
			textureProperties.height = textureHeight2;
			textureProperties.usageFlags = TextureUsageFlags::RenderTarget;
			GfxTexture* bloom4 = GfxTexturePool::Get(textureProperties);
			textureProperties.width = textureWidth2 / 2;
			textureProperties.height = textureHeight2 / 2;
			GfxTexture* bloom8 = GfxTexturePool::Get(textureProperties);
			textureProperties.width = textureWidth2 / 4;
			textureProperties.height = textureHeight2 / 4;
			GfxTexture* bloom16 = GfxTexturePool::Get(textureProperties);
			textureProperties.width = textureWidth2 / 8;
			textureProperties.height = textureHeight2 / 8;
			GfxTexture* bloom32 = GfxTexturePool::Get(textureProperties);

			GfxDevice::SetRenderTarget(bloom);
			GfxDevice::ClearColor({});
			GfxDevice::SetRenderTarget(nullptr);

			GfxDevice::SetGlobalTexture(s_MSAASourceTextureId, msaaColor);
			GfxDevice::SetGlobalTexture(s_ColorOutputTextureId, color);
			GfxDevice::SetGlobalTexture(s_BloomOutputTextureId, bloom);
			GfxDevice::Dispatch(s_ResolveMSAABloomShader->GetKernel(0), threadWidth, threadHeight, 1);
			AutoExposure::Calculate(camera, color, viewport);

			BloomData bloomConstants = {};
			bloomConstants.texelSize = Vector2(1.0f / textureWidth, 1.0f / textureHeight);
			s_BloomData->SetData(reinterpret_cast<char*>(&bloomConstants), sizeof(BloomData));

			GfxDevice::SetGlobalBuffer(s_BloomDataId, s_BloomData);

			// Horizontal blur
			GfxDevice::SetRenderTarget(bloom4);
			GfxDevice::SetViewport(0, 0, textureWidth, textureHeight);
			GfxDevice::SetGlobalTexture(s_SourceTextureId, bloom);
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_BloomMaterial, 0));

			bloomConstants.texelSize = Vector2(1.0f / textureWidth2, 1.0f / textureHeight2);
			s_BloomData->SetData(reinterpret_cast<char*>(&bloomConstants), sizeof(BloomData));

			// Horizontal blur
			GfxDevice::SetRenderTarget(bloom);
			GfxDevice::SetViewport(0, 0, textureWidth2, textureHeight2);
			GfxDevice::SetGlobalTexture(s_SourceTextureId, bloom4);
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_BloomMaterial, 0));

			bloomConstants.texelSize = Vector2(1.0f / textureWidth, 1.0f / textureHeight);
			s_BloomData->SetData(reinterpret_cast<char*>(&bloomConstants), sizeof(BloomData));

			// Vertical blur
			GfxDevice::SetRenderTarget(bloom4);
			GfxDevice::SetViewport(0, 0, textureWidth, textureHeight);
			GfxDevice::SetGlobalTexture(s_SourceTextureId, bloom);
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_BloomMaterial, 1));

			// Downscale 1
			GfxDevice::SetRenderTarget(bloom8);
			GfxDevice::SetViewport(0, 0, textureWidth2 / 2, textureHeight2 / 2);
			GfxDevice::SetGlobalTexture(s_SourceTextureId, bloom4);
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_BloomMaterial, 4));

			bloomConstants.texelSize = Vector2(1.0f / (textureWidth2 / 2), 1.0f / (textureHeight2 / 2));
			s_BloomData->SetData(reinterpret_cast<char*>(&bloomConstants), sizeof(BloomData));

			// Vertical blur
			GfxDevice::SetRenderTarget(bloom);
			GfxDevice::SetViewport(0, 0, textureWidth2 / 2, textureHeight2 / 2);
			GfxDevice::SetGlobalTexture(s_SourceTextureId, bloom8);
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_BloomMaterial, 3));

			bloomConstants.texelSize = Vector2(1.0f / textureWidth, 1.0f / textureHeight);
			s_BloomData->SetData(reinterpret_cast<char*>(&bloomConstants), sizeof(BloomData));

			// Horizontal blur
			GfxDevice::SetRenderTarget(bloom8);
			GfxDevice::SetViewport(0, 0, textureWidth, textureHeight);
			GfxDevice::SetGlobalTexture(s_SourceTextureId, bloom);
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_BloomMaterial, 2));

			// Downscale 2
			GfxDevice::SetRenderTarget(bloom16);
			GfxDevice::SetViewport(0, 0, textureWidth2 / 4, textureHeight2 / 4);
			GfxDevice::SetGlobalTexture(s_SourceTextureId, bloom8);
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_BloomMaterial, 4));

			bloomConstants.texelSize = Vector2(1.0f / (textureWidth2 / 4), 1.0f / (textureHeight2 / 4));
			s_BloomData->SetData(reinterpret_cast<char*>(&bloomConstants), sizeof(BloomData));
			
			// Vertical blur
			GfxDevice::SetRenderTarget(bloom);
			GfxDevice::SetViewport(0, 0, textureWidth2 / 4, textureHeight2 / 4);
			GfxDevice::SetGlobalTexture(s_SourceTextureId, bloom16);
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_BloomMaterial, 3));

			bloomConstants.texelSize = Vector2(1.0f / textureWidth, 1.0f / textureHeight);
			s_BloomData->SetData(reinterpret_cast<char*>(&bloomConstants), sizeof(BloomData));

			// Horizontal blur
			GfxDevice::SetRenderTarget(bloom16);
			GfxDevice::SetViewport(0, 0, textureWidth, textureHeight);
			GfxDevice::SetGlobalTexture(s_SourceTextureId, bloom);
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_BloomMaterial, 2));

			// Downscale 3
			GfxDevice::SetRenderTarget(bloom32);
			GfxDevice::SetViewport(0, 0, textureWidth2 / 8, textureHeight2 / 8);
			GfxDevice::SetGlobalTexture(s_SourceTextureId, bloom16);
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_BloomMaterial, 4));

			bloomConstants.texelSize = Vector2(1.0f / (textureWidth2 / 8), 1.0f / (textureHeight2 / 8));
			s_BloomData->SetData(reinterpret_cast<char*>(&bloomConstants), sizeof(BloomData));

			// Vertical blur
			GfxDevice::SetRenderTarget(bloom);
			GfxDevice::SetViewport(0, 0, textureWidth2 / 8, textureHeight2 / 8);
			GfxDevice::SetGlobalTexture(s_SourceTextureId, bloom32);
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_BloomMaterial, 3));

			bloomConstants.texelSize = Vector2(1.0f / textureWidth, 1.0f / textureHeight);
			s_BloomData->SetData(reinterpret_cast<char*>(&bloomConstants), sizeof(BloomData));

			// Horizontal blur
			GfxDevice::SetRenderTarget(bloom32);
			GfxDevice::SetViewport(0, 0, textureWidth, textureHeight);
			GfxDevice::SetGlobalTexture(s_SourceTextureId, bloom);
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_BloomMaterial, 2));

			bloomConstants.texelSize = Vector2(1.0f / (textureWidth2 / 8), 1.0f / (textureHeight2 / 8));
			s_BloomData->SetData(reinterpret_cast<char*>(&bloomConstants), sizeof(BloomData));

			// Upscale 3
			GfxDevice::SetRenderTarget(bloom16);
			GfxDevice::SetViewport(0, 0, textureWidth2 / 4, textureHeight2 / 4);
			GfxDevice::SetGlobalTexture(s_SourceTextureId, bloom32);
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_BloomMaterial, 5));

			bloomConstants.texelSize = Vector2(1.0f / (textureWidth2 / 4), 1.0f / (textureHeight2 / 4));
			s_BloomData->SetData(reinterpret_cast<char*>(&bloomConstants), sizeof(BloomData));

			// Upscale 2
			GfxDevice::SetRenderTarget(bloom8);
			GfxDevice::SetViewport(0, 0, textureWidth2 / 2, textureHeight2 / 2);
			GfxDevice::SetGlobalTexture(s_SourceTextureId, bloom16);
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_BloomMaterial, 5));

			bloomConstants.texelSize = Vector2(1.0f / (textureWidth2 / 2), 1.0f / (textureHeight2 / 2));
			s_BloomData->SetData(reinterpret_cast<char*>(&bloomConstants), sizeof(BloomData));

			// Upscale 1
			GfxDevice::SetRenderTarget(bloom);
			GfxDevice::SetViewport(0, 0, textureWidth2, textureHeight2);
			GfxDevice::SetGlobalTexture(s_SourceTextureId, bloom8);
			GfxDevice::SetGlobalTexture(s_SourceAdditionalTextureId, bloom4);
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_BloomMaterial, 6));
			GfxDevice::SetGlobalTexture(s_ScreenBloomTextureId, bloom);

			// Tonemapping
			// Gamma correction is done manually together with MSAA resolve to avoid using SRGB swapchain
			GfxDevice::SetRenderTarget(output);
			GfxDevice::SetViewport(0, 0, size.x, size.y);
			GfxDevice::SetGlobalTexture(s_ScreenColorTextureId, color);
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), DefaultMaterials::GetPostProcessing(), cameraType == CameraType::Reflection ? 1 : 0));
			GfxDevice::SetRenderTarget(nullptr);

			GfxTexturePool::Release(bloom);
			GfxTexturePool::Release(bloom4);
			GfxTexturePool::Release(bloom8);
			GfxTexturePool::Release(bloom16);
			GfxTexturePool::Release(bloom32);
		}
	}
}
