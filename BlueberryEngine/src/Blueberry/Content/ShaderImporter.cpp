#include "bbpch.h"
#include "ShaderImporter.h"

#include "Blueberry\Core\GlobalServices.h"

namespace Blueberry
{
	Ref<Object> ShaderImporter::Import(const std::string& path)
	{
		std::wstring shaderPath = std::wstring(path.begin(), path.end()).append(L".shader");
		return Shader::Create(shaderPath);
	}

	std::size_t ShaderImporter::GetType()
	{
		return Shader::Type;
	}
}