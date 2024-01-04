#include "bbpch.h"
#include "Texture.h"

#include "Blueberry\Graphics\GfxTexture.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Texture)

	const UINT& Texture::GetWidth()
	{
		return m_Width;
	}

	const UINT& Texture::GetHeight()
	{
		return m_Height;
	}

	void* Texture::GetHandle()
	{
		return m_Texture->GetHandle();
	}

	void Texture::BindProperties()
	{
	}
}