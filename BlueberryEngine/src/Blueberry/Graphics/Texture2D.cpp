#include "bbpch.h"
#include "Texture2D.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Texture, Texture2D)

	Texture2D::~Texture2D()
	{
		if (m_RawData.data != nullptr)
		{
			delete[] m_RawData.data;
		}
	}

	void Texture2D::Initialize(const TextureProperties& properties)
	{
		m_Width = properties.width;
		m_Height = properties.height;
		m_MipCount = properties.mipCount;
		m_Format = properties.format;
		m_WrapMode = properties.wrapMode;
		m_FilterMode = properties.filterMode;

		if (m_Texture != nullptr)
		{
			delete m_Texture;
		}

		GfxDevice::CreateTexture(properties, m_Texture);
	}

	void Texture2D::Initialize(const ByteData& byteData)
	{
		TextureProperties properties = {};

		properties.width = m_Width;
		properties.height = m_Height;
		properties.data = byteData.data;
		properties.dataSize = byteData.size;
		properties.mipCount = m_MipCount;
		properties.format = m_Format;
		properties.wrapMode = m_WrapMode;
		properties.filterMode = m_FilterMode;

		if (m_Texture != nullptr)
		{
			delete m_Texture;
		}

		GfxDevice::CreateTexture(properties, m_Texture);
	}

	Texture2D* Texture2D::Create(const TextureProperties& properties)
	{
		Texture2D* texture = Object::Create<Texture2D>();
		texture->Initialize(properties);
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
		BIND_FIELD(FieldInfo(TO_STRING(m_RawData), &Texture2D::m_RawData, BindingType::ByteData))
		END_OBJECT_BINDING()
	}
}