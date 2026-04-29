#include "EditorSerializer.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Serialization\YamlReader.h"
#include "Blueberry\Serialization\YamlWriter.h"
#include "Blueberry\Serialization\BinaryReader.h"
#include "Blueberry\Serialization\BinaryWriter.h"

#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Graphics\Texture3D.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\ComputeShader.h"
#include "Blueberry\Audio\AudioClip.h"
#include "Blueberry\Tools\FileHelper.h"

#include "Editor\Assets\AssetDB.h"
#include "Editor\Prefabs\PrefabInstance.h"
#include "Editor\Prefabs\PrefabManager.h"

#include "Editor\Assets\Processors\HLSLShaderProcessor.h"
#include "Editor\Assets\Processors\HLSLComputeShaderProcessor.h"
#include "Editor\Assets\Importers\ShaderImporter.h"
#include "Editor\Assets\Importers\ComputeShaderImporter.h"
#include "Editor\Assets\Importers\TextureImporter.h"
#include "Editor\Assets\Importers\AudioImporter.h"

#include <fstream>

namespace Blueberry
{
	void EditorSerializer::Serialize(const String& path, SerializationFlags flags)
	{
		bool isText = (flags & SerializationFlags::Text) != SerializationFlags::None;
		bool hasHeaders = (flags & SerializationFlags::HasHeaders) != SerializationFlags::None;
		bool hasGuids = (flags & SerializationFlags::HasGuids) != SerializationFlags::None;

		std::ofstream stream("temp", std::ios::out | std::ofstream::binary);
		if (stream.is_open())
		{
			for (ObjectId id : m_ObjectsToSerialize)
			{
				Object* object = ObjectDB::GetObject(id);
				if (object->IsClassType(PrefabInstance::Type))
				{
					if (!static_cast<PrefabInstance*>(object)->HasSource())
					{
						m_IsPrefabAsset = true;
						break;
					}
				}
			}

			while ((m_CurrentObject = GetNextObjectToSerialize()) != nullptr)
			{
				SerializationTree tree = {};
 				tree.typeId = m_CurrentObject->GetType();
				tree.fileId = GetFileId(m_CurrentObject->GetObjectId());
				tree.objectId = m_CurrentObject->GetObjectId();
				tree.isText = isText;
				if (PrefabManager::IsPartOfPrefabInstance(m_CurrentObject))
				{
					PrefabInstance* instance = PrefabManager::GetInstance(m_CurrentObject);
					if (instance->m_SourcePrefab.IsValid())
					{
						SerializationNodeRef rootNode = tree.GetRoot();
						PrefabInstance* instance = PrefabManager::GetInstance(m_CurrentObject);
						rootNode["m_PrefabInstance"] << GetPtrData(instance);
						rootNode["m_CorrespondingPrefabObject"] << GetPtrData(PrefabManager::GetCorrespondingPrefabObject(m_CurrentObject));
						tree.isReference = true;
					}
					else
					{
						SerializeNode(tree.GetRoot(), Context::Create(m_CurrentObject, m_CurrentObject->GetType()), flags);
					}
				}
				else
				{
					SerializeNode(tree.GetRoot(), Context::Create(m_CurrentObject, m_CurrentObject->GetType()), flags);
				}
				m_Trees.push_back(std::move(tree));
			}
			if (isText)
			{
				YamlWriter::Write(m_Trees, stream, hasHeaders);
			}
			else
			{
				BinaryWriter::Write(m_Trees, stream, hasGuids);
			}
			stream.close();
			std::filesystem::copy_file("temp", path, std::filesystem::copy_options::overwrite_existing);
			std::filesystem::remove("temp");
		}
	}

	void EditorSerializer::Deserialize(const String& path, SerializationFlags flags)
	{
		bool hasHeaders = (flags & SerializationFlags::HasHeaders) != SerializationFlags::None;
		bool hasGuids = (flags & SerializationFlags::HasGuids) != SerializationFlags::None;

		std::ifstream stream(path.data(), std::ios::in | std::ofstream::binary);
		if (stream.is_open())
		{
			m_Trees.clear();
			switch (stream.peek())
			{
			case '%':
				YamlReader::Read(m_Trees, stream, hasHeaders);
				break;
			case 'B':
				BinaryReader::Read(m_Trees, stream, hasGuids);
				break;
			}
			Guid invalidGuid = {};
			for (auto& tree : m_Trees)
			{
				Object* instance = nullptr;
				auto it = m_FileIdToObjectId.find(tree.fileId);
				if (it == m_FileIdToObjectId.end())
				{
					const ClassInfo* info = ClassDB::GetInfo(tree.typeId);
					if (info == nullptr)
					{
						BB_ERROR("Class not exists.");
						continue;
					}
					instance = info->Create();
				}
				else
				{
					instance = ObjectDB::GetObject(it->second);
				}
				tree.objectId = instance->GetObjectId();
				AddDeserializedObject(instance->GetObjectId(), hasGuids ? tree.guid : invalidGuid, tree.fileId);
			}
			for (auto& tree : m_Trees)
			{
				if (!tree.isReference)
				{
					Object* object = ObjectDB::GetObject(tree.objectId);
					if (object->IsClassType(PrefabInstance::Type))
					{
						PrefabInstance* instance = static_cast<PrefabInstance*>(object);
						m_PrefabInstances.push_back(instance);
					}
					DeserializeNode(tree.GetConstRoot(), Context::Create(object, object->GetType()));
				}
			}
			for (auto& tree : m_Trees)
			{
				if (tree.isReference)
				{
					SerializationNodeConstRef root = tree.GetConstRoot();
					ObjectPtrData prefabInstanceData = {};
					ObjectPtrData correspondingPrefabObjectData = {};
					root["m_PrefabInstance"] >> prefabInstanceData;
					root["m_CorrespondingPrefabObject"] >> correspondingPrefabObjectData;
					PrefabInstance* prefabInstance = static_cast<PrefabInstance*>(GetPtrObject(prefabInstanceData));
					Object* correspondingPrefabObject = GetPtrObject(correspondingPrefabObjectData);
					prefabInstance->AddObjectMapping(correspondingPrefabObject, ObjectDB::GetObject(tree.objectId));
					m_References.insert(tree.objectId);
				}
			}
			stream.close();
		}
	}

	void EditorSerializer::AddAdditionalObject(const ObjectId& objectId)
	{
		if (!m_IsPrefabAsset && PrefabManager::IsPartOfPrefabInstance(objectId) && (PrefabManager::IsPartOfPrefabInstance(m_CurrentObject) || m_CurrentObject->GetType() == PrefabInstance::Type))
		{
			return;
		}
		Serializer::AddAdditionalObject(objectId);
	}

	void EditorSerializer::GatherPrefabs(Scene* scene)
	{
		for (auto& rootEntity : scene->GetRootEntities())
		{
			GatherChildrenPrefabs(rootEntity.Get());
		}
	}

	void EditorSerializer::GatherDependencies(HashSet<Guid>& dependencies)
	{
		for (auto& tree : m_Trees)
		{
			for (size_t i = 0; i < tree.nodes.size(); ++i)
			{
				if (tree.nodes[i].key == "guid")
				{
					Guid guid;
					SerializationNodeConstRef ref = { i, &tree };
					ref >> guid;
					dependencies.insert(guid);
				}
			}
		}
	}

	void EditorSerializer::FinalizeObjects()
	{
		if (m_Guid.IsValid())
		{
			for (auto& tree : m_Trees)
			{
				Object* object = ObjectDB::GetObject(tree.objectId);
				Finalize(object, m_Guid, tree.fileId);
			}
		}
		if (m_PrefabInstances.size() > 0)
		{
			// Setup prefab hierarchy
			for (PrefabInstance* instance : m_PrefabInstances)
			{
				ObjectDB::AllocateIdToGuid(instance, m_Guid, ObjectDB::GetFileIdFromObject(instance));
				if (instance->HasSource())
				{
					instance->Initialize();
				}
			}
			// Components added to prefab
			for (auto& tree : m_Trees)
			{
				if (!tree.isReference)
				{
					Object* object = ObjectDB::GetObject(tree.objectId);
					if (object->IsClassType(Component::Type))
					{
						Component* component = static_cast<Component*>(object);
						Entity* entity = component->GetEntity();
						if (entity != nullptr && m_References.count(entity->GetObjectId()) > 0)
						{
							entity->AddComponent(component);
						}
					}
				}
			}
			// Setup prefab context
			for (PrefabInstance* instance : m_PrefabInstances)
			{
				if (!instance->HasSource())
				{
					instance->Initialize();
				}
			}
		}
	}

	void EditorSerializer::GatherChildrenPrefabs(Entity* entity)
	{
		if (entity == nullptr)
		{
			return;
		}
		if (PrefabManager::IsPartOfPrefabInstance(entity))
		{
			if (PrefabManager::IsPrefabInstanceRoot(entity))
			{
				PrefabInstance* instance = PrefabManager::GetInstance(entity);
				instance->PrepareData();
				AddObject(instance);
			}
			for (size_t i = 0; i < entity->GetComponentCount(); ++i)
			{
				Component* component = entity->GetComponentAt(i);
				if (!PrefabManager::IsPartOfPrefabInstance(component))
				{
					AddObject(component);
				}
			}
		}
		else
		{
			AddObject(entity);
			for (size_t i = 0; i < entity->GetComponentCount(); ++i)
			{
				AddObject(entity->GetComponentAt(i));
			}
		}
		Transform* transform = entity->GetTransform();
		for (auto& child : transform->GetChildren())
		{
			GatherChildrenPrefabs(child.Get()->GetEntity());
		}
	}

	void EditorSerializer::Finalize(Object* object, const Guid& guid, const FileId& fileId)
	{
		TypeId type = object->GetType();
		if (type == Mesh::Type)
		{
			Mesh* mesh = static_cast<Mesh*>(object);
			mesh->Apply();
		}
		else if (type == Texture2D::Type)
		{
			Texture2D* texture = static_cast<Texture2D*>(object);
			String texturePath = TextureImporter::GetTexturePath(guid);
			if (std::filesystem::exists(texturePath))
			{
				List<uint8_t> data = FileHelper::LoadBinary(texturePath);
				texture->SetData(data.data(), data.size());
			}
			texture->Apply();
		}
		else if (type == TextureCube::Type)
		{
			TextureCube* texture = static_cast<TextureCube*>(object);
			String texturePath = TextureImporter::GetTexturePath(guid);
			if (std::filesystem::exists(texturePath))
			{
				List<uint8_t> data = FileHelper::LoadBinary(texturePath);
				texture->SetData(data.data(), data.size());
			}
			texture->Apply();
		}
		else if (type == Texture3D::Type)
		{
			Texture3D* texture = static_cast<Texture3D*>(object);
			String texturePath = TextureImporter::GetTexturePath(guid);
			if (std::filesystem::exists(texturePath))
			{
				List<uint8_t> data = FileHelper::LoadBinary(texturePath);
				texture->SetData(data.data(), data.size());
			}
			texture->Apply();
		}
		else if (type == Shader::Type)
		{
			Shader* shader = static_cast<Shader*>(object);
			String folderPath = ShaderImporter::GetShaderFolder(guid);
			HLSLShaderProcessor processor;
			if (processor.LoadVariants(folderPath))
			{
				shader->Initialize(processor.GetVariantsData());
			}
		}
		else if (type == ComputeShader::Type)
		{
			ComputeShader* shader = static_cast<ComputeShader*>(object);
			String folderPath = ComputeShaderImporter::GetShaderFolder(guid);
			HLSLComputeShaderProcessor processor;
			if (processor.LoadKernels(folderPath))
			{
				shader->Initialize(processor.GetShaders());
			}
		}
		else if (type == AudioClip::Type)
		{
			AudioClip* audioClip = static_cast<AudioClip*>(object);
			String audioClipPath = AudioImporter::GetAudioPath(guid);
			List<uint8_t> data = FileHelper::LoadBinary(audioClipPath);
			audioClip->Initialize(data);
		}
		object->SetState(ObjectState::Default);
	}
}
