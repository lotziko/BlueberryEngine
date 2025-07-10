#include "CookieAtlas.h"

#include "..\RenderContext.h"
#include "..\LightHelper.h"
#include "Blueberry\Graphics\GfxRenderTexturePool.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\DefaultMaterials.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Scene\Components\Light.h"

namespace Blueberry
{
	static uint32_t s_MaxCookies = 16;
	List<ObjectId> CookieAtlas::s_Cookies = {};

	void CookieAtlas::Initialize()
	{
		s_AtlasTexture = GfxRenderTexturePool::Get(512, 512, s_MaxCookies, 1, TextureFormat::R8G8B8A8_UNorm_SRGB, TextureDimension::Texture3D, WrapMode::Clamp, FilterMode::Point);
		GfxDevice::SetRenderTarget(s_AtlasTexture);
		GfxDevice::ClearColor(Color(1, 1, 1, 1));
		GfxDevice::SetRenderTarget(nullptr);
		s_Cookies.emplace_back(0);
	}

	void CookieAtlas::PrepareCookies(CullingResults& results)
	{
		for (Light* light : results.lights)
		{
			switch (light->GetType())
			{
			case LightType::Directional:
				continue;
			case LightType::Point:
			case LightType::Spot:
				Texture* cookie = light->GetCookie();
				for (uint32_t i = 0; i < light->m_SliceCount; ++i)
				{
					uint32_t index = GetIndex(cookie);

					Matrix scaleBiasTransform = Matrix::Identity;
					scaleBiasTransform._11 = 0.5f;
					scaleBiasTransform._22 = -0.5f;
					scaleBiasTransform._33 = 0.0f;
					scaleBiasTransform._41 = 0.5f;
					scaleBiasTransform._42 = 0.5f;
					scaleBiasTransform._43 = (1.0f / (s_MaxCookies * 2)) + (float)index / s_MaxCookies;

					if (light->IsCastingShadows() && light->GetType() != LightType::Point)
					{
						light->m_WorldToCookie[i] = light->m_WorldToShadow[i] * scaleBiasTransform;
					}
					else
					{
						Matrix view = LightHelper::GetViewMatrix(light, light->GetTransform(), i);
						Matrix projection = LightHelper::GetProjectionMatrix(light);
						light->m_WorldToCookie[i] = view * projection * scaleBiasTransform;
					}
				}
				break;
			}
		}
	}

	GfxTexture* CookieAtlas::GetAtlasTexture()
	{
		return s_AtlasTexture;
	}

	uint32_t CookieAtlas::GetIndex(Texture* cookie)
	{
		if (cookie == nullptr)
		{
			return 0;
		}
		// Need to check UpdateCount too
		ObjectId id = cookie->GetObjectId();
		uint32_t cookieCount = static_cast<uint32_t>(s_Cookies.size());
		for (uint32_t i = 0; i < cookieCount; ++i)
		{
			if (id == s_Cookies[i])
			{
				return i;
			}
		}

		s_Cookies.emplace_back(id);
		GfxDevice::SetRenderTarget(s_AtlasTexture, nullptr, cookieCount);
		GfxDevice::SetViewport(0, 0, 512, 512);
		GfxDevice::SetGlobalTexture(TO_HASH("_BlitTexture"), cookie->Get());
		GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), DefaultMaterials::GetBlit()));
		GfxDevice::SetRenderTarget(nullptr);

		return cookieCount;
	}
}
