#pragma once

namespace Blueberry
{
	class GfxTexture
	{
	public:
		virtual ~GfxTexture() = default;

		virtual UINT GetWidth() const = 0;
		virtual UINT GetHeight() const = 0;
		virtual void* GetHandle() = 0;

		virtual void GetData(void* data) = 0;
		virtual void SetData(void* data) = 0;
	};
}