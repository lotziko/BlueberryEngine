#include "ProjectBuilder.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Serialization\Serializer.h"
#include "Blueberry\Serialization\BinaryWriter.h"
#include "Blueberry\Tools\FileHelper.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Graphics\Texture3D.h"
#include "Blueberry\Graphics\TextureCubeArray.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\ComputeShader.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Scene\LightingSettings.h"

#include "Editor\Assets\EditorAssetLoader.h"
#include "Editor\Assets\Importers\ShaderImporter.h"
#include "Editor\Assets\Importers\ComputeShaderImporter.h"
#include "Editor\Assets\Importers\TextureImporter.h"
#include "Editor\Assets\AssemblyManager.h"
#include "Editor\EditorSceneManager.h"
#include "Editor\Scene\SceneSettings.h"
#include "Editor\Scene\LightingData.h"

#include <fstream>

namespace Blueberry
{
	struct Context
	{
		Dictionary<Guid, size_t>& resourceBlobOffsets;
		std::ofstream& resourcesStream;
	};

	void WriteAssetResource(Context context, Object* object)
	{
		Guid guid = ObjectDB::GetGuidFromObject(object); // Maybe use fileId too?
		TypeId type = object->GetType();
		if (type == Texture2D::Type || type == TextureCube::Type || type == Texture3D::Type || type == TextureCubeArray::Type)
		{
			Texture* texture = static_cast<Texture*>(object);
			if (texture->HasData())
			{
				const ByteData& data = texture->GetData();
				size_t size = data.size();
				context.resourceBlobOffsets.insert_or_assign(guid, context.resourcesStream.tellp());
				context.resourcesStream.write(reinterpret_cast<char*>(&guid), sizeof(Guid));
				context.resourcesStream.write(reinterpret_cast<char*>(&size), sizeof(size_t));
				context.resourcesStream.write(reinterpret_cast<const char*>(data.data()), size);
			}
			else
			{
				String texturePath = TextureImporter::GetTexturePath(guid);
				if (std::filesystem::exists(texturePath))
				{
					List<uint8_t> data;
					FileHelper::Load(data, texturePath);
					size_t size = data.size();
					context.resourceBlobOffsets.insert_or_assign(guid, context.resourcesStream.tellp());
					context.resourcesStream.write(reinterpret_cast<char*>(&guid), sizeof(Guid));
					context.resourcesStream.write(reinterpret_cast<char*>(&size), sizeof(size_t));
					context.resourcesStream.write(reinterpret_cast<char*>(data.data()), size);
				}
			}
		}
		else if (type == Shader::Type)
		{
			String shaderFolder = ShaderImporter::GetShaderFolder(guid);
			if (std::filesystem::exists(shaderFolder))
			{
				String indexesPath = shaderFolder + "\\indexes";
				List<uint8_t> data;
				FileHelper::Load(data, indexesPath);
				size_t size = data.size();
				context.resourceBlobOffsets.insert_or_assign(guid, context.resourcesStream.tellp());
				context.resourcesStream.write(reinterpret_cast<char*>(&guid), sizeof(Guid));
				context.resourcesStream.write(reinterpret_cast<char*>(data.data()), size);

				uint32_t blobCount = *reinterpret_cast<uint32_t*>(data.data() + (size - sizeof(uint32_t)));
				for (uint32_t i = 0; i < blobCount; ++i)
				{
					String blobPath = shaderFolder + "\\" + String(std::to_string(i));
					if (std::filesystem::exists(blobPath))
					{
						data.clear();
						FileHelper::Load(data, blobPath);
						size = data.size();
						context.resourcesStream.write(reinterpret_cast<char*>(&size), sizeof(size_t));
						context.resourcesStream.write(reinterpret_cast<char*>(data.data()), size);
					}
				}
			}
		}
		else if (type == ComputeShader::Type)
		{
			String shaderFolder = ShaderImporter::GetShaderFolder(guid);
			if (std::filesystem::exists(shaderFolder))
			{
				String indexesPath = shaderFolder + "\\indexes";
				List<uint8_t> data;
				FileHelper::Load(data, indexesPath);
				context.resourceBlobOffsets.insert_or_assign(guid, context.resourcesStream.tellp());
				context.resourcesStream.write(reinterpret_cast<char*>(&guid), sizeof(Guid));

				uint32_t blobCount = *reinterpret_cast<uint32_t*>(data.data());
				context.resourcesStream.write(reinterpret_cast<char*>(&blobCount), sizeof(uint32_t));
				for (uint32_t i = 0; i < blobCount; ++i)
				{
					String blobPath = shaderFolder + "\\" + String(std::to_string(i));
					if (std::filesystem::exists(blobPath))
					{
						data.clear();
						FileHelper::Load(data, blobPath);
						size_t size = data.size();
						context.resourcesStream.write(reinterpret_cast<char*>(&size), sizeof(size_t));
						context.resourcesStream.write(reinterpret_cast<char*>(data.data()), size);
					}
				}
			}
		}
	}

	TextureCubeArray* CreateReflectionAtlas(LightingData* lightingData)
	{
		List<TextureCube*> probeTextures = lightingData->GetReflectionProbes();
		uint32_t probeCount = static_cast<uint32_t>(probeTextures.size());
		TextureCube* texture = probeTextures[0];
		size_t textureSize = texture->GetDataSize();
		List<uint8_t> textureData(textureSize * probeCount);
		for (uint32_t i = 0; i < probeCount; ++i)
		{
			TextureCube* texture = probeTextures[i];
			if (texture != nullptr)
			{
				memcpy(textureData.data() + textureSize * i, texture->GetData(), textureSize);
			}
		}
		uint32_t size = texture->GetWidth();
		TextureCubeArray* reflectionProbes = TextureCubeArray::Create(size, size, probeCount, texture->GetMipCount(), texture->GetFormat(), texture->GetWrapMode(), texture->GetFilterMode());
		reflectionProbes->SetData(textureData.data(), textureData.size());
		return reflectionProbes;
	}

	void GatherAssets(const ClassInfo* classInfo, void* ptr, HashSet<Object*>& visited, List<Object*>& stack)
	{
		for (auto& field : classInfo->fields)
		{
			switch (field.type)
			{
			case BindingType::ObjectPtr:
			{
				Object* object = field.Get<ObjectPtr<Object>>(ptr)->Get();
				if (object != nullptr && visited.count(object) == 0)
				{
					visited.insert(object);
					stack.push_back(object);
				}
			}
			break;
			case BindingType::ObjectPtrList:
			{
				List<ObjectPtr<Object>>* objectList = field.Get<List<ObjectPtr<Object>>>(ptr);
				for (size_t i = 0; i < objectList->size(); ++i)
				{
					Object* object = objectList->at(i).Get();
					if (object != nullptr && visited.count(object) == 0)
					{
						visited.insert(object);
						stack.push_back(object);
					}
				}
			}
			break;
			case BindingType::Data:
			{
				Data* data = field.Get<Data>(ptr);
				const ClassInfo* dataClassInfo = ClassDB::GetInfo(*field.options.objectType);
				if (dataClassInfo != nullptr)
				{
					GatherAssets(dataClassInfo, data, visited, stack);
				}
				else
				{
					BB_ERROR("Data class not exists.");
				}
			}
			break;
			case BindingType::DataList:
			{
				ListBase* dataArrayPointer = field.Get<ListBase>(ptr);
				const ClassInfo* dataClassInfo = ClassDB::GetInfo(*field.options.objectType);
				if (dataClassInfo != nullptr)
				{
					size_t dataSize = dataArrayPointer->size_base();
					for (size_t i = 0; i < dataSize; ++i)
					{
						void* data = dataArrayPointer->get_base(i);
						GatherAssets(dataClassInfo, data, visited, stack);
					}
				}
				else
				{
					BB_ERROR("Data class not exists.");
				}
			}
			break;
			}
		}
	}

	void ProjectBuilder::Build(Scene* scene, const String& path)
	{
		if (!std::filesystem::exists(path))
		{
			std::filesystem::create_directories(path);
		}

		// Game assembly
		if (AssemblyManager::BuildRuntime())
		{
			std::filesystem::copy_file(AssemblyManager::GetAssemblyDirectory() + "GameAssembly.dll", path + "\\GameAssembly.dll", std::filesystem::copy_options::overwrite_existing);
		}

		std::filesystem::copy_file("BlueberryRuntime.exe", path + "\\BlueberryRuntime.exe", std::filesystem::copy_options::overwrite_existing);
		std::filesystem::copy_file("GFSDK_SSAO_D3D11.win64.dll", path + "\\GFSDK_SSAO_D3D11.win64.dll", std::filesystem::copy_options::overwrite_existing);

		// Scene
		List<Object*> assetStack;
		Serializer sceneSerializer = {};
		LightingSettings* lightingSettings = nullptr;
		TextureCubeArray* reflectionProbes = nullptr;
		LightingData* lightingData = EditorSceneManager::GetSettings()->GetLightingData();
		if (lightingData != nullptr)
		{
			lightingSettings = Object::Create<LightingSettings>();
			lightingSettings->SetLightmap(lightingData->GetLightmap());
			lightingSettings->SetProbeVolume(lightingData->GetProbeVolume());
			lightingSettings->SetReflectionProbes(reflectionProbes = CreateReflectionAtlas(lightingData));
			ObjectDB::AllocateIdToGuid(reflectionProbes, Guid(TO_HASH("ReflectionProbes"), 0), 1);
			lightingSettings->SetChartOffsetScale(lightingData->GetChartOffsetScale());
			sceneSerializer.AddObject(lightingSettings);
			assetStack.push_back(lightingSettings);
		}

		for (auto& pair : scene->GetEntities())
		{
			Entity* entity = pair.second.Get();
			sceneSerializer.AddObject(entity);
			assetStack.push_back(entity);
		}
		sceneSerializer.Serialize(path + "\\Scene", SerializationFlags::RuntimeOnly);

		// Assets
		Serializer assetSerializer = {};
		HashSet<Object*> resourceObjects;
		HashSet<Object*> visitedAssets;
		while (assetStack.size() > 0)
		{
			Object* object = assetStack.back();
			assetStack.pop_back();
			GatherAssets(ClassDB::GetInfo(object->GetType()), object, visitedAssets, assetStack);
			if (ObjectDB::HasGuid(object))
			{
				assetSerializer.AddObject(object);
				resourceObjects.insert(object);
			}
		}

		// TODO put only engine assets
		EditorAssetLoader* assetLoader = static_cast<EditorAssetLoader*>(AssetLoader::GetInstance());
		for (const auto& entry : std::filesystem::recursive_directory_iterator("assets\\"))
		{
			std::filesystem::path path = entry.path();
			Object* object = AssetLoader::Load(StringHelper::ToGenericString(path));
			if (object != nullptr && ObjectDB::HasGuid(object))
			{
				assetSerializer.AddObject(object);
				resourceObjects.insert(object);
			}
		}
		assetSerializer.Serialize(path + "\\Assets", SerializationFlags::RuntimeOnly | SerializationFlags::HasGuids);

		// Resources
		std::ofstream resourcesStream((path + "\\Resources").data(), std::ios::out | std::ofstream::binary);
		if (resourcesStream.is_open())
		{
			Dictionary<Guid, size_t> resourceBlobOffsets;
			Context context = { resourceBlobOffsets, resourcesStream };
			for (Object* object : resourceObjects)
			{
				WriteAssetResource(context, object);
			}
			// TODO write physics shapes
			resourcesStream.close();
		}

		if (reflectionProbes != nullptr)
		{
			Object::Destroy(reflectionProbes);
		}
		if (lightingSettings != nullptr)
		{
			Object::Destroy(lightingSettings);
		}
	}
}