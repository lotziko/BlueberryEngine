#include "bbpch.h"
#include "Texture2D.h"

#include "Blueberry\Core\GlobalServices.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Texture, Texture2D)
	
	Texture2D::Texture2D(const TextureProperties& properties)
	{
		g_GraphicsDevice->CreateTexture(properties, m_Texture);
		m_Width = properties.width;
		m_Height = properties.height;
		m_RawData.data = reinterpret_cast<BYTE*>(properties.data);
		m_RawData.size = properties.dataSize;
	}

	Ref<Texture2D> Texture2D::Create(const TextureProperties& properties)
	{
		return ObjectDB::CreateObject<Texture2D>(properties);
	}

	void Texture2D::BindProperties()
	{
		BEGIN_OBJECT_BINDING(Texture2D)
		BIND_FIELD("m_Width", &Texture2D::m_Width, BindingType::Int)
		BIND_FIELD("m_Height", &Texture2D::m_Height, BindingType::Int)
		BIND_FIELD("m_RawData", &Texture2D::m_RawData, BindingType::ByteData)
		END_OBJECT_BINDING()
	}
}