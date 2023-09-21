#include "bbpch.h"
#include "Texture2D.h"

#include "Blueberry\Core\GlobalServices.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Texture, Texture2D)
	
	Texture2D::Texture2D(const std::string& path)
	{
		g_GraphicsDevice->CreateTexture(path, m_Texture);
	}

	Ref<Texture2D> Texture2D::Create(const std::string& path)
	{
		return ObjectDB::CreateObject<Texture2D>(path);
	}
}