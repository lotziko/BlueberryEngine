#include "CookieAtlas.h"

#include "..\RenderContext.h"
#include "..\LightHelper.h"
#include "..\GfxRenderTexturePool.h"
#include "..\GfxDevice.h"
#include "..\StandardMeshes.h"
#include "..\DefaultMaterials.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Scene\Components\Light.h"

namespace Blueberry
{
	static uint32_t s_MaxCookies = 16;
	List<ObjectId> CookieAtlas::s_Cookies = {};

	void CookieAtlas::Initialize()
	{
		s_AtlasTexture = GfxRenderTexturePool::Get(512, 512, s_MaxCookies, 1, TextureFormat::R8G8B8A8_UNorm_SRGB, TextureDimension::Texture3D, WrapMode::Clamp, FilterMode::Point);
	}

	void CookieAtlas::PrepareCookies(CullingResults& results)
	{
		for (Light* light : results.lights)
		{
			if (light->GetType() != LightType::Spot)
			{
				continue;
			}

			Texture* cookie = light->GetCookie();
			if (cookie != nullptr)
			{
				uint32_t index = GetIndex(cookie);

				Matrix scaleBiasTransform = Matrix::Identity;
				scaleBiasTransform._11 = 0.5f;
				scaleBiasTransform._22 = -0.5f;
				scaleBiasTransform._33 = 0.0f;
				scaleBiasTransform._41 = 0.5f;
				scaleBiasTransform._42 = 0.5f;
				scaleBiasTransform._43 = (1.0f / (s_MaxCookies * 2)) + (float)index / s_MaxCookies;

				// Not sure if this is right, maybe cookie should have different matrix in point lights at least
				if (light->IsCastingShadows())
				{
					light->m_WorldToCookie[0] = light->m_WorldToShadow[0] * scaleBiasTransform;
				}
				else
				{
					Matrix view = LightHelper::GetViewMatrix(light);
					Matrix projection = LightHelper::GetProjectionMatrix(light);
					light->m_WorldToCookie[0] = view * projection * scaleBiasTransform;
				}
			}
		}
	}

	GfxTexture* CookieAtlas::GetAtlasTexture()
	{
		return s_AtlasTexture;
	}

	uint32_t CookieAtlas::GetIndex(Texture* cookie)
	{
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
