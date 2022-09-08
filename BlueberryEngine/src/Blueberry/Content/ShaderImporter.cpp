#include "bbpch.h"
#include "ShaderImporter.h"

#include "Blueberry\Core\GlobalServices.h"

namespace Blueberry
{
	Ref<Object> ShaderImporter::Import(const std::string& path)
	{
		static Ref<Shader> ref;
		std::wstring vertexPath = std::wstring(path.begin(), path.end()).append(L"-v.cso");
		std::wstring pixelPath = std::wstring(path.begin(), path.end()).append(L"-p.cso");
		g_GraphicsDevice->CreateShader(vertexPath, pixelPath, ref);
		return ref;
	}

	std::size_t ShaderImporter::GetType()
	{
		return Shader::Type;
	}
}