#include "bbpch.h"
#include "EditorAssetLoader.h"

#include <filesystem>
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\AssetImporter.h"
#include "Editor\ShaderProcessor.h"
#include "Blueberry\Graphics\Shader.h"

namespace Blueberry
{
	Object* EditorAssetLoader::LoadImpl(const Guid& guid, const FileId& fileId)
	{
		AssetImporter* importer = AssetDB::Import(guid);
		if (importer != nullptr)
		{
			auto importedObjects = importer->GetImportedObjects();
			auto objectIt = importedObjects.find(fileId);
			if (objectIt != importedObjects.end())
			{
				return ObjectDB::IdToObjectItem(objectIt->second)->object;
			}
		}
		return nullptr;
	}

	Object* EditorAssetLoader::LoadImpl(const std::string& path)
	{
		std::filesystem::path assetPath = path;
		std::string extension = assetPath.extension().string();
		if (extension == ".shader")
		{
			std::string shaderData;
			RawShaderOptions options;
			ShaderProcessor::Process(path, shaderData, options);
			void* vertex = ShaderProcessor::Compile(shaderData, "Vertex", "vs_5_0", "");
			void* fragment = ShaderProcessor::Compile(shaderData, "Fragment", "ps_5_0", "");
			return Shader::Create(vertex, fragment, options);
		}
		else if (extension == ".compute")
		{
			void* compute = ShaderProcessor::Compile(path, "Main", "cs_5_0", "");
		}
		return nullptr;
	}
}