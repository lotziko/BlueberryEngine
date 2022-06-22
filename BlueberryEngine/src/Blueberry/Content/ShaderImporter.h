#pragma once

#include "Blueberry\Content\AssetImporter.h"
#include "Blueberry\Graphics\Shader.h"

namespace Blueberry
{
	class ShaderImporter : public AssetImporter
	{
	public:
		virtual Ref<Object> Import(const std::string& path) final;
		virtual std::size_t GetType() final { return Shader::Type; }
	};
}