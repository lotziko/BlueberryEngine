#include "bbpch.h"
#include "EditorAssetLoader.h"

#include <filesystem>

#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Texture2D.h"

#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\AssetImporter.h"
#include "Editor\Assets\Processors\HLSLShaderParser.h"
#include "Editor\Assets\Processors\HLSLShaderProcessor.h"
#include "Editor\Assets\Processors\PngTextureProcessor.h"

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
			std::string shaderCode;
			RawShaderOptions options;

			if (HLSLShaderParser::Parse(path, shaderCode, options))
			{
				HLSLShaderProcessor vertexProcessor;
				HLSLShaderProcessor fragmentProcessor;

				vertexProcessor.Compile(shaderCode, ShaderType::Vertex);
				fragmentProcessor.Compile(shaderCode, ShaderType::Fragment);

				return Shader::Create(vertexProcessor.GetShader(), fragmentProcessor.GetShader(), options);
			}
			return nullptr;
		}
		else if (extension == ".png")
		{
			PngTextureProcessor processor;
			processor.Load(path);
			return Texture2D::Create(processor.GetProperties());
		}
		else if (extension == ".compute")
		{
		}
		return nullptr;
	}
}