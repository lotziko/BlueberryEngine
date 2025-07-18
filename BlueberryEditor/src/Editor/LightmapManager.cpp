#include "LightmapManager.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\DefaultTextures.h"
#include "Blueberry\Graphics\Texture2D.h"

#include "Baking\LightmappingManager.h"
#include "Editor\EditorSceneManager.h"

namespace Blueberry
{
	static GfxTexture* s_LightmappingRenderTarget = nullptr;

	void LightmapManager::Initialize()
	{
		GfxDevice::SetGlobalTexture(TO_HASH("_LightmapTexture"), DefaultTextures::GetWhite2D()->Get());
	}

	void LightmapManager::Bake()
	{
		try
		{
			if (s_LightmappingRenderTarget == nullptr)
			{
				TextureProperties properties = {};
				properties.width = 1024;
				properties.height = 1024;
				properties.depth = 1;
				properties.antiAliasing = 1;
				properties.format = TextureFormat::R32G32B32A32_Float;
				properties.dimension = TextureDimension::Texture2D;
				properties.wrapMode = WrapMode::Clamp;
				properties.filterMode = FilterMode::Bilinear;
				properties.isWritable = true;

				GfxDevice::CreateTexture(properties, s_LightmappingRenderTarget);

				size_t frameBufferSize = 1024 * 1024 * sizeof(Vector4);
				uint8_t* result = BB_MALLOC_ARRAY(uint8_t, frameBufferSize);
				LightmappingManager::Calculate(EditorSceneManager::GetScene(), Vector2Int(1024, 1024), result);
				s_LightmappingRenderTarget->SetData(result, frameBufferSize);
				GfxDevice::SetGlobalTexture(TO_HASH("_LightmapTexture"), s_LightmappingRenderTarget);
				BB_FREE(result);
				LightmappingManager::Shutdown();
			}
		}
		catch (...)
		{
		}
	}

	GfxTexture* LightmapManager::GetTexture()
	{
		return s_LightmappingRenderTarget;
	}
}
