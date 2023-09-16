#include "bbpch.h"
#include "Texture.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Texture)

	void* Texture::GetHandle() 
	{ 
		return NULL; 
	}

	void Texture::Bind() const
	{
	}
}