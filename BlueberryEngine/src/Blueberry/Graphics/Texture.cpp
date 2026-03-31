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
		DEFINE_FIELD(Texture, m_Width, BindingType::Int, FieldOptions())
		DEFINE_FIELD(Texture, m_Height, BindingType::Int, FieldOptions())
		DEFINE_FIELD(Texture, m_MipCount, BindingType::Int, FieldOptions())
		DEFINE_FIELD(Texture, m_Format, BindingType::Enum, FieldOptions())
		DEFINE_FIELD(Texture, m_WrapMode, BindingType::Enum, FieldOptions().SetEnumHint("Repeat,Clamp"))
		DEFINE_FIELD(Texture, m_FilterMode, BindingType::Enum, FieldOptions().SetEnumHint("Linear,Point,Trilinear"))
		DEFINE_FIELD(Texture, m_RawData, BindingType::ByteData, FieldOptions())
		DEFINE_FIELD(Texture, m_IsReadable, BindingType::Bool, FieldOptions())
	}

	uint32_t Texture::GetWidth() const
	{
		return m_Width;
	}

	uint32_t Texture::GetHeight() const
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

	WrapMode Texture::GetWrapMode() const
	{
		return m_WrapMode;
	}

	void Texture::SetWrapMode(WrapMode wrapMode)
	{
		m_WrapMode = wrapMode;
		if (m_Texture != nullptr)
		{
			m_Texture->SetWrapMode(wrapMode);
		}
	}

	FilterMode Texture::GetFilterMode() const
	{
		return m_FilterMode;
	}

	void Texture::SetFilterMode(FilterMode filterMode)
	{
		m_FilterMode = filterMode;
		if (m_Texture != nullptr)
		{
			m_Texture->SetFilterMode(filterMode);
		}
	}

	bool Texture::IsReadable() const
	{
		return m_IsReadable;
	}

	void Texture::SetReadable(bool readable)
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