#include "EditorAssetLoader.h"

#include <filesystem>

#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\ComputeShader.h"

#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\AssetImporter.h"
#include "Editor\Assets\Processors\PngTextureProcessor.h"
#include "Editor\Assets\Processors\HLSLShaderProcessor.h"
#include "Editor\Assets\Processors\HLSLComputeShaderProcessor.h"

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
			return ObjectDB::GetObjectFromGuid(guid, fileId);
		}
		return nullptr;
	}

	Object* EditorAssetLoader::LoadImpl(const String& path)
	{
		auto it = m_LoadedAssets.find(path);
		if (it != m_LoadedAssets.end())
		{
			return it->second;
		}

		std::filesystem::path assetPath = path;
		std::string extension = assetPath.extension().string();
		if (extension == ".png")
		{
			PngTextureProcessor processor;
			processor.Load(path, false);
			PngTextureProperties properties = processor.GetProperties();
			Texture2D* texture = Texture2D::Create(properties.width, properties.height, properties.mipCount, properties.format, WrapMode::Repeat);
			texture->SetName(assetPath.stem().string().data());
			texture->SetData(static_cast<uint8_t*>(properties.data), properties.dataSize);
			texture->Apply();
			m_LoadedAssets.insert_or_assign(path, texture);
			return texture;
		}
		else if (extension == ".shader")
		{
			std::string shaderCode;
			HLSLShaderProcessor processor;
			if (processor.Compile(path))
			{
				Shader* shader = Shader::Create(processor.GetVariantsData(), processor.GetShaderData());
				shader->SetName(assetPath.stem().string().data());
				m_LoadedAssets.insert_or_assign(path, shader);
				return shader;
			}
			return nullptr;
		}
		else if (extension == ".compute")
		{
			std::string shaderCode;
			HLSLComputeShaderProcessor processor;
			if (processor.Compile(path))
			{
				ComputeShader* shader = ComputeShader::Create(processor.GetShaders(), processor.GetComputeShaderData());
				shader->SetName(assetPath.stem().string().data());
				m_LoadedAssets.insert_or_assign(path, shader);
				return shader;
			}
			return nullptr;
		}
		return nullptr;
	}
}