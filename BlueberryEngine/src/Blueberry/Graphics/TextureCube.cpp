#include "bbpch.h"
#include "TextureCube.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Texture, TextureCube)

	TextureCube::~TextureCube()
	{
		if (m_Texture != nullptr)
		{
			delete m_Texture;
		}
	}

	void TextureCube::SetData(uint8_t* data, const size_t& dataSize)
	{
		m_RawData.data = data;
		m_RawData.size = dataSize;
	}

	void TextureCube::Apply()
	{
		TextureProperties textureProperties = {};

		textureProperties.width = m_Width;
		textureProperties.height = m_Height;
		textureProperties.data = m_RawData.data;
		textureProperties.dataSize = m_RawData.size;
		textureProperties.mipCount = m_MipCount;
		textureProperties.format = m_Format;
		textureProperties.dimension = TextureDimension::TextureCube;
		textureProperties.wrapMode = m_WrapMode;
		textureProperties.filterMode = m_FilterMode;
	
		if (m_Texture != nullptr)
		{
			delete m_Texture;
		}

		GfxDevice::CreateTexture(textureProperties, m_Texture);
	}

	TextureCube* TextureCube::Create(const uint32_t& width, const uint32_t& height, const uint32_t& mipCount, const TextureFormat& textureFormat, const WrapMode& wrapMode, const FilterMode& filterMode, TextureCube* existingTexture)
	{
		TextureCube* texture = nullptr;
		if (existingTexture != nullptr)
		{
			texture = existingTexture;
		}
		else
		{
			texture = Object::Create<TextureCube>();
		}
		texture->m_Width = width;
		texture->m_Height = height;
		texture->m_MipCount = mipCount;
		texture->m_Format = textureFormat;
		texture->m_WrapMode = wrapMode;
		texture->m_FilterMode = filterMode;
		return texture;
	}

	void TextureCube::BindProperties()
	{
		BEGIN_OBJECT_BINDING(TextureCube)
		BIND_FIELD(FieldInfo(TO_STRING(m_Width), &TextureCube::m_Width, BindingType::Int))
		BIND_FIELD(FieldInfo(TO_STRING(m_Height), &TextureCube::m_Height, BindingType::Int))
		BIND_FIELD(FieldInfo(TO_STRING(m_MipCount), &TextureCube::m_MipCount, BindingType::Int))
		BIND_FIELD(FieldInfo(TO_STRING(m_Format), &TextureCube::m_Format, BindingType::Enum))
		BIND_FIELD(FieldInfo(TO_STRING(m_WrapMode), &TextureCube::m_WrapMode, BindingType::Enum).SetHintData("Repeat,Clamp"))
		BIND_FIELD(FieldInfo(TO_STRING(m_FilterMode), &TextureCube::m_FilterMode, BindingType::Enum).SetHintData("Linear,Point"))
		//BIND_FIELD(FieldInfo(TO_STRING(m_RawData), &Texture2D::m_RawData, BindingType::ByteData))
		END_OBJECT_BINDING()
	}
}