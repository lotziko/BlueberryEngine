#include "Blueberry\Graphics\TextureCube.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(TextureCube, Texture)
	{
		DEFINE_BASE_FIELDS(TextureCube, Texture)
	}

	TextureCube::~TextureCube()
	{
		if (m_Texture != nullptr)
		{
			delete m_Texture;
		}
	}

	uint8_t* TextureCube::GetData()
	{
		return m_RawData.data();
	}

	const size_t TextureCube::GetDataSize()
	{
		return m_RawData.size();
	}

	void TextureCube::Initialize(const uint32_t& width, const uint32_t& height, const uint32_t& mipCount, const TextureFormat& textureFormat)
	{
		m_Width = width;
		m_Height = height;
		m_MipCount = mipCount;
		m_Format = textureFormat;
		if (m_Texture != nullptr)
		{
			delete m_Texture;
			m_Texture = nullptr;
		}
	}

	void TextureCube::SetData(uint8_t* data, const size_t& dataSize)
	{
		m_RawData.resize(dataSize);
		memcpy(m_RawData.data(), data, dataSize);
	}

	void TextureCube::Apply()
	{
		if (m_Texture != nullptr)
		{
			delete m_Texture;
			m_Texture = nullptr;
		}

		TextureProperties textureProperties = {};

		textureProperties.width = m_Width;
		textureProperties.height = m_Height;
		textureProperties.data = m_RawData.data();
		textureProperties.dataSize = m_RawData.size();
		textureProperties.mipCount = m_MipCount;
		textureProperties.format = m_Format;
		textureProperties.dimension = TextureDimension::TextureCube;
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

	TextureCube* TextureCube::Create(const uint32_t& width, const uint32_t& height, const uint32_t& mipCount, const TextureFormat& textureFormat, const WrapMode& wrapMode, const FilterMode& filterMode)
	{
		TextureCube* texture = Object::Create<TextureCube>();
		texture->m_Width = width;
		texture->m_Height = height;
		texture->m_MipCount = mipCount;
		texture->m_Format = textureFormat;
		texture->m_WrapMode = wrapMode;
		texture->m_FilterMode = filterMode;
		return texture;
	}
}