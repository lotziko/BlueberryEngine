#include "EditorAssetLoader.h"

#include <filesystem>

#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\ComputeShader.h"
#include "Blueberry\Tools\FileHelper.h"

#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\AssetImporter.h"
#include "Editor\Assets\Importers\TextureImporter.h"
#include "Editor\Assets\Importers\ShaderImporter.h"
#include "Editor\Assets\Importers\ComputeShaderImporter.h"
#include "Editor\Assets\Processors\HLSLShaderProcessor.h"
#include "Editor\Assets\Processors\HLSLComputeShaderProcessor.h"
#include "Editor\Misc\TextureHelper.h"
#include "Editor\Misc\PathHelper.h"

#include <directxtex\DirectXTex.h>

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
		std::string name = assetPath.filename().string();
		Guid guid = Guid(TO_HASH(String(name)), 0);
		std::string extension = assetPath.extension().string();
		
		if (extension == ".png")
		{
			String texturePath = TextureImporter::GetTexturePath(guid);
			Texture2D* texture = nullptr;

			bool needImport = true;
			if (AssetDB::HasAssetWithGuidInData(guid) && std::filesystem::exists(texturePath.data()) && PathHelper::GetLastWriteTime(path) < PathHelper::GetLastWriteTime(texturePath))
			{
				auto objects = AssetDB::LoadAssetObjects(guid, ObjectDB::GetObjectsFromGuid(guid));
				if (objects.size() == 1 && objects[0].first->IsClassType(Texture2D::Type))
				{
					texture = static_cast<Texture2D*>(objects[0].first);
					ObjectDB::AllocateIdToGuid(texture, guid, 1);
					uint8_t* data;
					size_t length;
					FileHelper::Load(data, length, texturePath);
					texture->SetData(data, length);
					texture->Apply();
					needImport = false;
				}
			}

			if (needImport)
			{
				DirectX::ScratchImage image = {};
				TextureHelper::Load(image, path, ".png", true);
				TextureHelper::Flip(image);
				auto metadata = image.GetMetadata();
				texture = Texture2D::Create(metadata.width, metadata.height, metadata.mipLevels, static_cast<TextureFormat>(metadata.format), WrapMode::Repeat);
				ObjectDB::AllocateIdToGuid(texture, guid, 1);
				texture->SetData(static_cast<uint8_t*>(image.GetPixels()), image.GetPixelsSize());
				texture->Apply();

				AssetDB::SaveAssetObjectsToCache(List<Object*> { texture });
				FileHelper::Save(image.GetPixels(), image.GetPixelsSize(), texturePath);
			}

			if (texture != nullptr)
			{
				texture->SetName(assetPath.stem().string().data());
				m_LoadedAssets.insert_or_assign(path, texture);
			}
			return texture;
		}
		else if (extension == ".shader")
		{
			String folderPath = ShaderImporter::GetShaderFolder(guid);
			auto folderWriteTime = PathHelper::GetDirectoryLastWriteTime(folderPath);
			Shader* shader = nullptr;
			HLSLShaderProcessor processor = {};

			bool needImport = true;
			if (AssetDB::HasAssetWithGuidInData(guid) && PathHelper::GetLastWriteTime(path) < folderWriteTime && ShaderImporter::GetLastFilesWriteTime() < folderWriteTime && processor.LoadVariants(folderPath))
			{
				auto objects = AssetDB::LoadAssetObjects(guid, ObjectDB::GetObjectsFromGuid(guid));
				if (objects.size() == 1 && objects[0].first->IsClassType(Shader::Type))
				{
					shader = static_cast<Shader*>(objects[0].first);
					ObjectDB::AllocateIdToGuid(shader, guid, 1);
					shader->Initialize(processor.GetVariantsData());
					needImport = false;
				}
			}

			if (needImport)
			{
				processor = {};
				if (processor.Compile(path))
				{
					processor.SaveVariants(folderPath);
					shader = Shader::Create(processor.GetVariantsData(), processor.GetShaderData());
					ObjectDB::AllocateIdToGuid(shader, guid, 1);
					AssetDB::SaveAssetObjectsToCache(List<Object*> { shader });
				}
			}

			if (shader != nullptr)
			{
				shader->SetName(assetPath.stem().string().data());
				m_LoadedAssets.insert_or_assign(path, shader);
			}
			return shader;
		}
		else if (extension == ".compute")
		{
			String folderPath = ComputeShaderImporter::GetShaderFolder(guid);
			auto folderWriteTime = PathHelper::GetDirectoryLastWriteTime(folderPath);
			ComputeShader* shader = nullptr;
			HLSLComputeShaderProcessor processor = {};

			bool needImport = true;
			if (AssetDB::HasAssetWithGuidInData(guid) && PathHelper::GetLastWriteTime(path) < folderWriteTime && ShaderImporter::GetLastFilesWriteTime() < folderWriteTime && processor.LoadKernels(folderPath))
			{
				auto objects = AssetDB::LoadAssetObjects(guid, ObjectDB::GetObjectsFromGuid(guid));
				if (objects.size() == 1 && objects[0].first->IsClassType(ComputeShader::Type))
				{
					shader = static_cast<ComputeShader*>(objects[0].first);
					ObjectDB::AllocateIdToGuid(shader, guid, 1);
					shader->Initialize(processor.GetShaders());
					needImport = false;
				}
			}
			
			if (needImport)
			{
				processor = {};
				if (processor.Compile(path))
				{
					processor.SaveKernels(folderPath);
					shader = ComputeShader::Create(processor.GetShaders(), processor.GetComputeShaderData());
					ObjectDB::AllocateIdToGuid(shader, guid, 1);
					AssetDB::SaveAssetObjectsToCache(List<Object*> { shader });
				}
			}

			if (shader != nullptr)
			{
				shader->SetName(assetPath.stem().string().data());
				m_LoadedAssets.insert_or_assign(path, shader);
			}
			return shader;
		}
		return nullptr;
	}
}