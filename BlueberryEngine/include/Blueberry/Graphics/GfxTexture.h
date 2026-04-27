#pragma once

#include "Blueberry\Core\Base.h"

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

		virtual uint32_t GetWidth() const = 0;
		virtual uint32_t GetHeight() const = 0;
		virtual TextureFormat GetFormat() const = 0;
		virtual void* GetHandle() = 0;

		virtual void GetData(void* target, const Rectangle& area) = 0;
		virtual void GetData(void* target) = 0;
		virtual void SetData(void* data, size_t size) = 0;
		virtual void SetData(void* data, size_t size, uint32_t slice) = 0;

		virtual void SetWrapMode(WrapMode wrapMode) = 0;
		virtual void SetFilterMode(FilterMode filterMode) = 0;
		virtual void SetName(const String& name) = 0;

		virtual void GenerateMipMaps() = 0;

	protected:
		uint32_t m_Index;

		friend class Material;
		friend class Texture;
	};
}