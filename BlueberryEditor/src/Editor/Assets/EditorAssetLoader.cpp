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
	void EditorAssetLoader::LoadImpl(const Guid& guid)
	{
		AssetImporter* importer = AssetDB::GetImporter(guid);
		if (importer != nullptr)
		{
			importer->ImportDataIfNeeded();
		}
	}

	Object* EditorAssetLoader::LoadImpl(const Guid& guid, const FileId& fileId)
	{
		AssetImporter* importer = AssetDB::GetImporter(guid);
		if (importer != nullptr)
		{
			importer->ImportDataIfNeeded();
			auto importedObjects = importer->GetImportedObjects();
			auto objectIt = importedObjects.find(fileId);
			if (objectIt != importedObjects.end())
			{
				return ObjectDB::GetObject(objectIt->second);
			}
		}
		return nullptr;
	}

	Object* EditorAssetLoader::LoadImpl(const std::string& path)
	{
		auto it = m_LoadedAssets.find(path);
		if (it != m_LoadedAssets.end())
		{
			return it->second;
		}

		std::filesystem::path assetPath = path;
		std::string extension = assetPath.extension().string();
		if (extension == ".shader")
		{
			std::string shaderCode;
			ShaderData data;

			if (HLSLShaderParser::Parse(path, shaderCode, data))
			{
				HLSLShaderProcessor vertexProcessor;
				HLSLShaderProcessor fragmentProcessor;

				vertexProcessor.Compile(shaderCode, ShaderType::Vertex);
				fragmentProcessor.Compile(shaderCode, ShaderType::Fragment);

				Shader* shader = Shader::Create(vertexProcessor.GetShader(), fragmentProcessor.GetShader(), data);
				m_LoadedAssets.insert_or_assign(path, shader);
				return shader;
			}
			return nullptr;
		}
		else if (extension == ".png")
		{
			PngTextureProcessor processor;
			processor.Load(path, false, false);
			TextureProperties properties = processor.GetProperties();
			Texture2D* texture = Texture2D::Create(properties);
			m_LoadedAssets.insert_or_assign(path, texture);
			return texture;
		}
		else if (extension == ".compute")
		{
		}
		return nullptr;
	}
}