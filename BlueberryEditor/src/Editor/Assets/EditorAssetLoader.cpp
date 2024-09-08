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
			HLSLShaderProcessor processor;
			if (processor.Compile(path))
			{
				Shader* shader = Shader::Create(processor.GetVariantsData(), processor.GetShaderData());
				m_LoadedAssets.insert_or_assign(path, shader);
				return shader;
			}
			return nullptr;
		}
		else if (extension == ".png")
		{
			PngTextureProcessor processor;
			processor.Load(path, false, false);
			PngTextureProperties properties = processor.GetProperties();
			Texture2D* texture = Texture2D::Create(properties.width, properties.height, properties.mipCount, properties.format);
			texture->SetName(assetPath.filename().string());
			texture->SetData((byte*)properties.data, properties.dataSize);
			texture->Apply();
			m_LoadedAssets.insert_or_assign(path, texture);
			return texture;
		}
		else if (extension == ".compute")
		{
		}
		return nullptr;
	}
}