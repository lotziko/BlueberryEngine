#include "bbpch.h"
#include "Texture2D.h"

#include "Blueberry\Core\GlobalServices.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Texture, Texture2D)
	
	Texture2D::Texture2D(const TextureProperties& properties)
	{
		g_GraphicsDevice->CreateTexture(properties, m_Texture);
	}

	Ref<Texture2D> Texture2D::Create(const TextureProperties& properties)
	{
		return ObjectDB::CreateObject<Texture2D>(properties);
	}
}