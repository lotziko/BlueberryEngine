#pragma once

namespace Blueberry
{
	class GfxTexture
	{
	public:
		BB_OVERRIDE_NEW_DELETE
		
		virtual ~GfxTexture() = default;

		virtual uint32_t GetWidth() const = 0;
		virtual uint32_t GetHeight() const = 0;
		virtual void* GetHandle() = 0;

		virtual void GetData(void* target, const Rectangle& area) = 0;
		virtual void GetData(void* target) = 0;
		virtual void SetData(void* data, const size_t& size) = 0;

		virtual void GenerateMipMaps() = 0;
	};
}