#include "bbpch.h"
#include "Texture.h"

#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Core\ObjectDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Texture)

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

	void Texture::BindProperties()
	{
	}
}