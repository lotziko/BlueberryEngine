#include "bbpch.h"
#include "Texture2D.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Texture, Texture2D)

	void Texture2D::Initialize(const ByteData& byteData)
	{
		GfxDevice::CreateTexture({ m_Width, m_Height, byteData.data, byteData.size, TextureType::Resource, TextureFormat::R8G8B8A8_UNorm }, m_Texture);
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