#include "Blueberry\Graphics\TextureCubeArray.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(TextureCubeArray, Texture)
	{
		DEFINE_BASE_FIELDS(TextureCubeArray, Texture)
		DEFINE_FIELD(TextureCubeArray, m_Count, BindingType::Int, FieldOptions())
	}

	TextureCubeArray::~TextureCubeArray()
	{
		if (m_Texture != nullptr)
		{
			delete m_Texture;
		}
	}

	uint32_t TextureCubeArray::GetCount() const
	{
		return m_Count;
	}

	void TextureCubeArray::SetData(uint8_t* data, size_t dataSize)
	{
		m_RawData.resize(dataSize);
		memcpy(m_RawData.data(), data, dataSize);
	}

	void TextureCubeArray::Apply()
	{
		if (m_Texture != nullptr)
		{
			delete m_Texture;
			m_Texture = nullptr;
		}

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
		IncrementUpdateCount();
		if (!m_IsReadable)
		{
			m_RawData.clear();
			m_RawData.shrink_to_fit();
		}
	}

	TextureCubeArray* TextureCubeArray::Create(uint32_t width, uint32_t height, uint32_t count, uint32_t mipCount, TextureFormat textureFormat, WrapMode wrapMode, FilterMode filterMode)
	{
		TextureCubeArray* texture = Object::Create<TextureCubeArray>();
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