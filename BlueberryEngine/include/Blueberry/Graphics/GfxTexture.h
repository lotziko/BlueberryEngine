#pragma once

#include "Blueberry\Core\Base.h"
#include "Enums.h"

namespace Blueberry
{
	enum class TextureFormat;
	enum class WrapMode;
	enum class FilterMode;

	class GfxTexture
	{
	public:
		BB_OVERRIDE_NEW_DELETE
		
		virtual ~GfxTexture() = default;

		uint32_t GetWidth() const;
		uint32_t GetHeight() const;
		uint32_t GetDepth() const;
		TextureFormat GetFormat() const;
		uint32_t GetAntiAliasing() const;
		uint32_t GetQuality() const;
		uint32_t GetArraySize() const;
		uint32_t GetMipLevels() const;
		WrapMode GetWrapMode() const;
		FilterMode GetFilterMode() const;

		uint32_t GetIndex() const;

		virtual void* GetHandle() = 0;

		virtual void GetData(void* data, const Rectangle& area) = 0;
		virtual void GetData(void* data) = 0;
		virtual void SetData(void* data, size_t size) = 0;

		virtual size_t GetAllocationSize() = 0;

		virtual void SetWrapMode(WrapMode wrapMode) = 0;
		virtual void SetFilterMode(FilterMode filterMode) = 0;
		virtual void SetName(const String& name) = 0;

	protected:
		TextureFormat m_Format = TextureFormat::None;
		uint32_t m_Width = 0;
		uint32_t m_Height = 0;
		uint32_t m_Depth = 0;
		uint32_t m_AntiAliasing = 1;
		uint32_t m_Quality = 0;
		uint32_t m_ArraySize = 0;
		uint32_t m_MipLevels = 1;
		TextureDimension m_Dimension = TextureDimension::Texture2D;
		WrapMode m_WrapMode = WrapMode::Clamp;
		FilterMode m_FilterMode = FilterMode::Bilinear;
		uint32_t m_Index = 0;
		size_t m_AllocationSize = UINT64_MAX;
	};
}