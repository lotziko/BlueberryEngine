#include "Blueberry\Graphics\Texture.h"

#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Core\Notifyable.h"
#include "Blueberry\Graphics\GfxTexture.h"
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
		DEFINE_FIELD(Texture, m_IsReadable, BindingType::Bool, {})
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

	const WrapMode& Texture::GetWrapMode()
	{
		return m_WrapMode;
	}

	void Texture::SetWrapMode(const WrapMode& wrapMode)
	{
		m_WrapMode = wrapMode;
		if (m_Texture != nullptr)
		{
			m_Texture->SetWrapMode(wrapMode);
		}
	}

	const FilterMode& Texture::GetFilterMode()
	{
		return m_FilterMode;
	}

	void Texture::SetFilterMode(const FilterMode& filterMode)
	{
		m_FilterMode = filterMode;
		if (m_Texture != nullptr)
		{
			m_Texture->SetFilterMode(filterMode);
		}
	}

	const bool& Texture::IsReadable()
	{
		return m_IsReadable;
	}

	void Texture::SetReadable(const bool& readable)
	{
		m_IsReadable = readable;
	}

	void Texture::IncrementUpdateCount()
	{
		++m_UpdateCount;
		for (auto dependency : m_Dependencies)
		{
			Object* object = ObjectDB::GetObject(dependency);
			if (object != nullptr)
			{
				dynamic_cast<Notifyable*>(object)->OnNotify(static_cast<Object*>(this));
			}
		}
	}
}