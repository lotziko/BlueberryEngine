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

	void Texture2D::Initialize(const ByteData& byteData)
	{
		TextureProperties properties = {};

		properties.width = m_Width;
		properties.height = m_Height;
		properties.data = byteData.data;
		properties.dataSize = byteData.size;
		properties.format = TextureFormat::R8G8B8A8_UNorm;

		GfxDevice::CreateTexture(properties, m_Texture);
	}

	Texture2D* Texture2D::Create(const TextureProperties& properties)
	{
		Texture2D* texture = Object::Create<Texture2D>();
		texture->m_Width = properties.width;
		texture->m_Height = properties.height;
		texture->Initialize({ (byte*)properties.data, properties.dataSize });
		return texture;
	}

	void Texture2D::BindProperties()
	{
		BEGIN_OBJECT_BINDING(Texture2D)
		BIND_FIELD(FieldInfo(TO_STRING(m_Width), &Texture2D::m_Width, BindingType::Int))
		BIND_FIELD(FieldInfo(TO_STRING(m_Height), &Texture2D::m_Height, BindingType::Int))
		BIND_FIELD(FieldInfo(TO_STRING(m_RawData), &Texture2D::m_RawData, BindingType::ByteData))
		END_OBJECT_BINDING()
	}
}