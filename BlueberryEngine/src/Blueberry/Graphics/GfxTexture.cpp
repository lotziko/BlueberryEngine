#include "Blueberry\Graphics\GfxTexture.h"

namespace Blueberry
{
	uint32_t GfxTexture::GetWidth() const
	{
		return m_Width;
	}

	uint32_t GfxTexture::GetHeight() const
	{
		return m_Height;
	}

	uint32_t GfxTexture::GetDepth() const
	{
		return m_Depth;
	}

	TextureFormat GfxTexture::GetFormat() const
	{
		return m_Format;
	}

	uint32_t GfxTexture::GetAntiAliasing() const
	{
		return m_AntiAliasing;
	}

	uint32_t GfxTexture::GetQuality() const
	{
		return m_Quality;
	}

	uint32_t GfxTexture::GetArraySize() const
	{
		return m_ArraySize;
	}

	uint32_t GfxTexture::GetMipLevels() const
	{
		return m_MipLevels;
	}

	WrapMode GfxTexture::GetWrapMode() const
	{
		return m_WrapMode;
	}

	FilterMode GfxTexture::GetFilterMode() const
	{
		return m_FilterMode;
	}

	uint32_t GfxTexture::GetIndex() const
	{
		return m_Index;
	}
}