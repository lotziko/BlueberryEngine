#include "Blueberry\Graphics\Texture2D.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Texture2D, Texture)
	{
		DEFINE_BASE_FIELDS(Texture2D, Texture)
	}

	Texture2D::~Texture2D()
	{
		if (m_Texture != nullptr)
		{
			delete m_Texture;
		}
		if (m_RawData.data != nullptr)
		{
			BB_FREE(m_RawData.data);
		}
	}

	void Texture2D::SetData(uint8_t* data, const size_t& dataSize)
	{
		m_RawData.data = data;
		m_RawData.size = dataSize;
	}

	void Texture2D::Apply()
	{
		TextureProperties textureProperties = {};

		textureProperties.width = m_Width;
		textureProperties.height = m_Height;
		textureProperties.data = m_RawData.data;
		textureProperties.dataSize = m_RawData.size;
		textureProperties.mipCount = m_MipCount;
		textureProperties.format = m_Format;
		textureProperties.dimension = TextureDimension::Texture2D;
		textureProperties.wrapMode = m_WrapMode;
		textureProperties.filterMode = m_FilterMode;

		if (m_Texture != nullptr)
		{
			delete m_Texture;
		}

		GfxDevice::CreateTexture(textureProperties, m_Texture);
	}

	Texture2D* Texture2D::Create(const uint32_t& width, const uint32_t& height, const uint32_t& mipCount, const TextureFormat& textureFormat, const WrapMode& wrapMode, const FilterMode& filterMode, Texture2D* existingTexture)
	{
		Texture2D* texture = nullptr;
		if (existingTexture != nullptr)
		{
			texture = existingTexture;
			texture->IncrementUpdateCount();
		}
		else
		{
			texture = Object::Create<Texture2D>();
		}
		texture->m_Width = width;
		texture->m_Height = height;
		texture->m_MipCount = mipCount;
		texture->m_Format = textureFormat;
		texture->m_WrapMode = wrapMode;
		texture->m_FilterMode = filterMode;
		return texture;
	}
}