#include "Blueberry\Graphics\TextureCubeArray.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(TextureCubeArray, Texture)
	{
		DEFINE_BASE_FIELDS(TextureCubeArray, Texture)
		DEFINE_FIELD(TextureCubeArray, m_Count, BindingType::Int, {})
	}

	TextureCubeArray::~TextureCubeArray()
	{
		if (m_Texture != nullptr)
		{
			delete m_Texture;
		}
	}

	const uint32_t& TextureCubeArray::GetCount()
	{
		return m_Count;
	}

	void TextureCubeArray::SetData(uint8_t* data, const size_t& dataSize)
	{
		m_RawData.resize(dataSize);
		memcpy(m_RawData.data(), data, dataSize);
	}

	void TextureCubeArray::Apply()
	{
		TextureProperties textureProperties = {};

		textureProperties.width = m_Width;
		textureProperties.height = m_Height;
		textureProperties.depth = m_Count;
		textureProperties.data = m_RawData.data();
		textureProperties.dataSize = m_RawData.size();
		textureProperties.mipCount = m_MipCount;
		textureProperties.format = m_Format;
		textureProperties.dimension = TextureDimension::TextureCubeArray;
		textureProperties.wrapMode = m_WrapMode;
		textureProperties.filterMode = m_FilterMode;

		GfxDevice::CreateTexture(textureProperties, m_Texture);
	}

	TextureCubeArray* TextureCubeArray::Create(const uint32_t& width, const uint32_t& height, const uint32_t& count, const uint32_t& mipCount, const TextureFormat& textureFormat, const WrapMode& wrapMode, const FilterMode& filterMode, TextureCubeArray* existingTexture)
	{
		TextureCubeArray* texture = nullptr;
		if (existingTexture != nullptr)
		{
			texture = existingTexture;
			texture->IncrementUpdateCount();
		}
		else
		{
			texture = Object::Create<TextureCubeArray>();
		}
		texture->m_Width = width;
		texture->m_Height = height;
		texture->m_Count = count;
		texture->m_MipCount = mipCount;
		texture->m_Format = textureFormat;
		texture->m_WrapMode = wrapMode;
		texture->m_FilterMode = filterMode;
		return texture;
	}
}