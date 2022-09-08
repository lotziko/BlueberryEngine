#include "bbpch.h"
#include "TextureImporter.h"

#include "Blueberry\Core\GlobalServices.h"

namespace Blueberry
{
	Ref<Object> TextureImporter::Import(const std::string& path)
	{
		static Ref<Texture> ref;
		std::string texturePath = std::string(path).append(".png");
		g_GraphicsDevice->CreateTexture(texturePath, ref);
		return ref;
	}

	std::size_t TextureImporter::GetType()
	{
		return Texture::Type;
	}
}