#pragma once

#include "Blueberry\Content\AssetImporter.h"
#include "Blueberry\Graphics\Texture.h"

namespace Blueberry
{
	class TextureImporter : public AssetImporter
	{
	public:
		virtual Ref<Object> Import(const std::string& path) final;
		virtual std::size_t GetType() final { return Texture::Type; }
	};
}