#include "bbpch.h"
#include "ShaderImporter.h"

#include "Blueberry\Core\GlobalServices.h"

namespace Blueberry
{
	Ref<Object> ShaderImporter::Import(const std::string& path)
	{
		static Ref<Shader> ref;
		std::wstring shaderPath = std::wstring(path.begin(), path.end()).append(L".hlsl");
		g_GraphicsDevice->CreateShader(shaderPath, ref);
		return ref;
	}

	std::size_t ShaderImporter::GetType()
	{
		return Shader::Type;
	}
}