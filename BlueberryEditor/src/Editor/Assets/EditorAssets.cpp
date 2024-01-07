#include "bbpch.h"
#include "EditorAssets.h"
#include <filesystem>

#include "Editor\ShaderProcessor.h"
#include "Blueberry\Graphics\Shader.h"

namespace Blueberry
{
	Object* EditorAssets::Load(const std::string& path)
	{
		std::filesystem::path assetPath = path;
		if (assetPath.extension() == ".shader")
		{
			void* vertex = ShaderProcessor::Compile(path, "Vertex", "vs_5_0", "");
			void* fragment = ShaderProcessor::Compile(path, "Fragment", "ps_5_0", "");
			return Shader::Create(vertex, fragment);
		}
		return nullptr;
	}
}