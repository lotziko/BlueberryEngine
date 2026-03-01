#include "ShadowAtlas.h"

#include "..\RenderContext.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\DefaultMaterials.h"
#include "..\LightHelper.h"
#include "Blueberry\Graphics\GfxTexturePool.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Light.h"
#include "Blueberry\Assets\AssetLoader.h"

namespace Blueberry
{
	#define SIZE 4096

	Material* ShadowAtlas::s_ShadowAtlasMaterial = nullptr;
	GfxTexture* ShadowAtlas::s_AtlasTexture = nullptr;
	GfxBuffer* ShadowAtlas::s_DepthBlitData = nullptr;
	List<ShadowAtlas::ShadowRequest> ShadowAtlas::s_Requests = {};

	static size_t s_DepthBlitDataId = TO_HASH("DepthBlitData");
	static size_t s_BlitTextureId = TO_HASH("_BlitTexture");

	struct DepthBlitData
	{
		Vector4 offsetScale;
	};

	void ShadowAtlas::Initialize()
	{
		s_ShadowAtlasMaterial = Material::Create(static_cast<Shader*>(AssetLoader::Load("assets/shaders/ShadowAtlas.shader")));

		TextureProperties textureProperties = {};

		textureProperties.width = SIZE;
		textureProperties.height = SIZE;
		textureProperties.depth = 1;
		textureProperties.antiAliasing = 1;
		textureProperties.mipCount = 1;
		textureProperties.format = TextureFormat::D32_Float;
		textureProperties.dimension = TextureDimension::Texture2D;
		textureProperties.wrapMode = WrapMode::Clamp;
		textureProperties.filterMode = FilterMode::CompareDepth;
		textureProperties.usageFlags = TextureUsageFlags::RenderTarget;

		GfxDevice::CreateTexture(textureProperties, s_AtlasTexture);

		BufferProperties depthBlitBufferProperties = {};
		depthBlitBufferProperties.elementCount = 1;
		depthBlitBufferProperties.elementSize = sizeof(DepthBlitData) * 1;
		depthBlitBufferProperties.usageFlags = BufferUsageFlags::ConstantBuffer;

		GfxDevice::CreateBuffer(depthBlitBufferProperties, s_DepthBlitData);
	}

	void ShadowAtlas::Shutdown()
	{
		delete s_AtlasTexture;
		delete s_DepthBlitData;
	}

	void ShadowAtlas::Clear()
	{
		s_Requests.clear();
	}

	void ShadowAtlas::Insert(Light* light)
	{
		LightType type = light->GetType();
		uint32_t size = LightHelper::GetShadowSize(type);
		uint32_t sliceCount = LightHelper::GetSliceCount(type);

		for (uint32_t i = 0; i < sliceCount; ++i)
		{
			ShadowRequest request = { size, 0u, 0u, light, i, sliceCount };
			s_Requests.push_back(request);
		}
	}

	void ShadowAtlas::Draw(RenderContext& context, CullingResults& results)
	{
		PackRequests();

		GfxDevice::SetDepthBias(0, 2.5f);
		GfxDevice::SetRenderTarget(nullptr, s_AtlasTexture);
		GfxDevice::ClearDepth(1.0f);

		// If has dirty flag render cached slices
		// Switching isStatic should update shadows

		for (size_t i = 0; i < s_Requests.size(); ++i)
		{
			ShadowRequest request = s_Requests[i];
			uint32_t sliceIndex = request.sliceIndex;
			float sliceScale = 1.0f / static_cast<float>(request.sliceCount);
			uint32_t size = request.size;
			uint32_t offset = size * sliceIndex;
			Light* light = request.light;

			ShadowDrawingSettings shadowDrawingSettings = {};
			shadowDrawingSettings.light = light;
			shadowDrawingSettings.sliceIndex = request.sliceIndex;

			if (light->m_IsCached)
			{
				GfxTexture* cachedTexture = light->GetCachedShadow();

				if (light->m_IsDirty[sliceIndex])
				{
					shadowDrawingSettings.objectsFilter = ObjectsFilter::Static;

					GfxDevice::SetRenderTarget(nullptr, cachedTexture);
					GfxDevice::SetViewport(offset, 0, size, size);
					GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_ShadowAtlasMaterial, 0));
					context.DrawShadows(results, shadowDrawingSettings);
					GfxDevice::SetRenderTarget(nullptr, s_AtlasTexture);

					light->m_IsDirty[sliceIndex] = false;
				}

				shadowDrawingSettings.objectsFilter = ObjectsFilter::Dynamic;
				GfxDevice::SetViewport(request.offsetX, request.offsetY, request.size, request.size);

				DepthBlitData depthBlitConstants = {};
				depthBlitConstants.offsetScale = Vector4(sliceIndex * sliceScale, 0, sliceScale, 1);
				s_DepthBlitData->SetData(reinterpret_cast<char*>(&depthBlitConstants), sizeof(DepthBlitData));

				GfxDevice::SetGlobalBuffer(s_DepthBlitDataId, s_DepthBlitData);
				GfxDevice::SetGlobalTexture(s_BlitTextureId, cachedTexture);
				GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_ShadowAtlasMaterial, 1));
			}
			else
			{
				shadowDrawingSettings.objectsFilter = ObjectsFilter::All;
				GfxDevice::SetViewport(request.offsetX, request.offsetY, request.size, request.size);
			}

			context.DrawShadows(results, shadowDrawingSettings);

			Matrix sliceTransform = Matrix::Identity;
			sliceTransform._11 = static_cast<float>(request.size) / SIZE;
			sliceTransform._22 = static_cast<float>(request.size) / SIZE;
			sliceTransform._41 = static_cast<float>(request.offsetX) / SIZE;
			sliceTransform._42 = static_cast<float>(request.offsetY) / SIZE;

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
		GfxDevice::SetRenderTarget(nullptr);
	}

	const Vector2Int ShadowAtlas::GetSize()
	{
		return Vector2Int(SIZE, SIZE);
	}

	GfxTexture* ShadowAtlas::GetAtlasTexture()
	{
		return s_AtlasTexture;
	}

	bool CompareRequests(ShadowAtlas::ShadowRequest s1, ShadowAtlas::ShadowRequest s2)
	{
		return s1.size > s2.size;
	}

	void ShadowAtlas::PackRequests()
	{
		std::sort(s_Requests.begin(), s_Requests.end(), CompareRequests);

		int layer = 0;
		int free = 1;
		Vector2Int layerSize = Vector2Int(SIZE, SIZE);
		Vector2Int previousSize = layerSize;
		for (size_t i = 0; i < s_Requests.size(); ++i)
		{
			uint32_t currentRequestSize = s_Requests[i].size;

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
					s_Requests[i].offsetX += layerSize.x >> (j + 1);
				}
				if ((slotIndex & (1 << (2 * (layerDepth - j - 1) + 1))) != 0)
				{
					s_Requests[i].offsetY += layerSize.y >> (j + 1);
				}
			}

			previousSize = Vector2Int(currentRequestSize, currentRequestSize);
			--free;
		}
	}
}
