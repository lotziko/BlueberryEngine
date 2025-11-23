#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	enum class TextureFormat;

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
		virtual void SetData(void* data, const size_t& size) = 0;
		virtual void SetData(void* data, const size_t& size, const uint32_t& slice) = 0;

		virtual void GenerateMipMaps() = 0;

	protected:
		uint32_t m_Index;

		friend class Material;
	};
}