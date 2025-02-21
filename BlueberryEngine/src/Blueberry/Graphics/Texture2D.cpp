#include "bbpch.h"
#include "Texture2D.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Texture, Texture2D)

	Texture2D::~Texture2D()
	{
		if (m_Texture != nullptr)
		{
			delete m_Texture;
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

	void Texture2D::BindProperties()
	{
		BEGIN_OBJECT_BINDING(Texture2D)
		BIND_FIELD(FieldInfo(TO_STRING(m_Width), &Texture2D::m_Width, BindingType::Int))
		BIND_FIELD(FieldInfo(TO_STRING(m_Height), &Texture2D::m_Height, BindingType::Int))
		BIND_FIELD(FieldInfo(TO_STRING(m_MipCount), &Texture2D::m_MipCount, BindingType::Int))
		BIND_FIELD(FieldInfo(TO_STRING(m_Format), &Texture2D::m_Format, BindingType::Enum))
		BIND_FIELD(FieldInfo(TO_STRING(m_WrapMode), &Texture2D::m_WrapMode, BindingType::Enum).SetHintData("Repeat,Clamp"))
		BIND_FIELD(FieldInfo(TO_STRING(m_FilterMode), &Texture2D::m_FilterMode, BindingType::Enum).SetHintData("Linear,Point"))
		//BIND_FIELD(FieldInfo(TO_STRING(m_RawData), &Texture2D::m_RawData, BindingType::ByteData))
		END_OBJECT_BINDING()
	}
}