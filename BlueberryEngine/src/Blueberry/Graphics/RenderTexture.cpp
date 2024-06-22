#include "bbpch.h"
#include "RenderTexture.h"

#include "Blueberry\Graphics\GfxDevice.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Texture, RenderTexture)

	RenderTexture::~RenderTexture()
	{
		if (m_Texture != nullptr)
		{
			delete m_Texture;
		}
	}

	RenderTexture* RenderTexture::Create(const UINT& width, const UINT& height, const UINT& antiAliasing, const TextureFormat& textureFormat, const WrapMode& wrapMode, const FilterMode& filterMode)
	{
		RenderTexture* texture = Object::Create<RenderTexture>();
		texture->m_Width = width;
		texture->m_Height = height;
		texture->m_MipCount = 1;
		texture->m_Format = textureFormat;
		texture->m_WrapMode = wrapMode;
		texture->m_FilterMode = filterMode;

		TextureProperties textureProperties = {};

		textureProperties.width = width;
		textureProperties.height = height;
		textureProperties.antiAliasing = antiAliasing;
		textureProperties.mipCount = 1;
		textureProperties.format = textureFormat;
		textureProperties.wrapMode = wrapMode;
		textureProperties.filterMode = filterMode;
		textureProperties.isRenderTarget = true;

		GfxDevice::CreateTexture(textureProperties, texture->m_Texture);

		return texture;
	}
}