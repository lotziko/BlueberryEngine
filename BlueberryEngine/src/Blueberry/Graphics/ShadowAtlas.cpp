#include "bbpch.h"
#include "ShadowAtlas.h"

#include "Blueberry\Graphics\RenderContext.h"
#include "Blueberry\Graphics\RenderTexture.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\LightHelper.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Light.h"

namespace Blueberry
{
	ShadowAtlas::ShadowAtlas(const uint32_t& width, const uint32_t& height, const uint32_t& maxLightCount) : m_MaxLightCount(maxLightCount)
	{
		m_AtlasTexture = RenderTexture::Create(width, height, 1, 1, TextureFormat::D32_Float, TextureDimension::Texture2D, WrapMode::Clamp, FilterMode::CompareDepth);
		m_Requests = BB_MALLOC_ARRAY(ShadowRequest, maxLightCount);
		m_Size = Vector2Int(width, height);
	}

	ShadowAtlas::~ShadowAtlas()
	{
		BB_FREE(m_Requests);
	}

	void ShadowAtlas::Clear()
	{
		m_RequestCount = 0;
	}

	void ShadowAtlas::Insert(Light* light, const uint32_t& size, const uint8_t& sliceCount)
	{
		if (light->GetType() != LightType::Point)
		{
			
		}

		for (int i = 0; i < sliceCount; ++i)
		{
			ShadowRequest request = { size, 0, 0, light, i };
			m_Requests[m_RequestCount] = request;
			++m_RequestCount;
		}

		light->m_SliceCount = sliceCount;
	}

	void ShadowAtlas::Draw(RenderContext& context, CullingResults& results)
	{
		PackRequests();

		GfxDevice::SetDepthBias(1, 2.5f);
		GfxDevice::SetRenderTarget(nullptr, m_AtlasTexture->Get());
		GfxDevice::ClearDepth(1.0f);

		for (int i = 0; i < m_RequestCount; ++i)
		{
			ShadowRequest request = m_Requests[i];
			uint8_t sliceIndex = request.sliceIndex;
			Light* light = request.light;

			ShadowDrawingSettings shadowDrawingSettings = {};
			shadowDrawingSettings.light = light;
			shadowDrawingSettings.sliceIndex = request.sliceIndex;

			GfxDevice::SetViewport(request.offsetX, request.offsetY, request.size, request.size);
			context.DrawShadows(results, shadowDrawingSettings);

			Matrix sliceTransform = Matrix::Identity;
			sliceTransform._11 = static_cast<float>(request.size) / m_AtlasTexture->GetWidth();
			sliceTransform._22 = static_cast<float>(request.size) / m_AtlasTexture->GetHeight();
			sliceTransform._41 = static_cast<float>(request.offsetX) / m_AtlasTexture->GetWidth();
			sliceTransform._42 = static_cast<float>(request.offsetY) / m_AtlasTexture->GetHeight();

			Matrix scaleBiasTransform = Matrix::Identity;
			scaleBiasTransform._11 = 0.5f;
			scaleBiasTransform._22 = -0.5f;
			scaleBiasTransform._41 = 0.5f;
			scaleBiasTransform._42 = 0.5f;
			scaleBiasTransform._43 = 0.0f;

			light->m_AtlasWorldToShadow[sliceIndex] = light->m_WorldToShadow[sliceIndex] * scaleBiasTransform * sliceTransform;
			light->m_ShadowBounds[sliceIndex] = Vector4(sliceTransform._41, sliceTransform._42, sliceTransform._41 + sliceTransform._11, sliceTransform._42 + sliceTransform._22);
		}

		GfxDevice::SetDepthBias(0, 0);
	}

	const Vector2Int& ShadowAtlas::GetSize()
	{
		return m_Size;
	}

	RenderTexture* ShadowAtlas::GetAtlasTexture()
	{
		return m_AtlasTexture;
	}

	bool CompareRequests(ShadowAtlas::ShadowRequest s1, ShadowAtlas::ShadowRequest s2)
	{
		return s1.size > s2.size;
	}

	void ShadowAtlas::PackRequests()
	{
		std::sort(m_Requests, m_Requests + m_RequestCount, CompareRequests);

		struct Vector2Int
		{
			uint32_t x;
			uint32_t y;
		};

		int layer = 0;
		int free = 1;
		Vector2Int layerSize = { m_AtlasTexture->GetWidth(), m_AtlasTexture->GetHeight() };
		Vector2Int previousSize = layerSize;
		for (uint32_t i = 0; i < m_RequestCount; ++i)
		{
			uint32_t currentRequestSize = m_Requests[i].size;

			if (free == 0)
			{
				++layer;
				free = 1;
				previousSize = layerSize;
			}

			free *= (previousSize.x / currentRequestSize * previousSize.y / currentRequestSize);

			uint32_t sideSlotCount = layerSize.x / currentRequestSize;
			uint32_t layerDepth = static_cast<uint32_t>(log2(sideSlotCount));
			uint32_t slotIndex = sideSlotCount * sideSlotCount - free;

			for (uint32_t j = 0; j < layerDepth; ++j)
			{
				if ((slotIndex & (1 << 2 * (layerDepth - j - 1))) != 0)
				{
					m_Requests[i].offsetX += layerSize.x >> (j + 1);
				}
				if ((slotIndex & (1 << (2 * (layerDepth - j - 1) + 1))) != 0)
				{
					m_Requests[i].offsetY += layerSize.y >> (j + 1);
				}
			}

			previousSize = { currentRequestSize, currentRequestSize };
			--free;
		}
	}
}
