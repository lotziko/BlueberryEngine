#include "bbpch.h"
#include "TextureImporter.h"

#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Core\GlobalServices.h"

namespace Blueberry
{
	Ref<Object> TextureImporter::Import(const std::string& path)
	{
		std::string texturePath = std::string(path).append(".png");
		return Texture2D::Create(texturePath);
	}

	std::size_t TextureImporter::GetType()
	{
		return Texture2D::Type;
	}
}