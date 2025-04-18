#include "bbpch.h"
#include "Texture.h"

#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Texture, Object)
	{
		DEFINE_BASE_FIELDS(Texture, Object)
		DEFINE_FIELD(Texture, m_Width, BindingType::Int, {})
		DEFINE_FIELD(Texture, m_Height, BindingType::Int, {})
		DEFINE_FIELD(Texture, m_MipCount, BindingType::Int, {})
		DEFINE_FIELD(Texture, m_Format, BindingType::Enum, {})
		DEFINE_FIELD(Texture, m_WrapMode, BindingType::Enum, FieldOptions().SetEnumHint("Repeat,Clamp"))
		DEFINE_FIELD(Texture, m_FilterMode, BindingType::Enum, FieldOptions().SetEnumHint("Linear,Point"))
	}

	const uint32_t& Texture::GetWidth()
	{
		return m_Width;
	}

	const uint32_t& Texture::GetHeight()
	{
		return m_Height;
	}

	GfxTexture* Texture::Get()
	{
		return m_Texture;
	}

	void* Texture::GetHandle()
	{
		if (m_Texture != nullptr)
		{
			return m_Texture->GetHandle();
		}
		return nullptr;
	}
}