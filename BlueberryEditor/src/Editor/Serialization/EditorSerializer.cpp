#include "EditorSerializer.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Serialization\YamlReader.h"
#include "Blueberry\Serialization\YamlWriter.h"
#include "Blueberry\Serialization\BinaryReader.h"
#include "Blueberry\Serialization\BinaryWriter.h"

#include "Editor\Assets\AssetDB.h"
#include "Editor\Prefabs\PrefabInstance.h"
#include "Editor\Prefabs\PrefabManager.h"

#include <fstream>

namespace Blueberry
{
	void EditorSerializer::Serialize(const String& path, const bool& isText)
	{
		std::ofstream stream(path.data(), std::ios::out | std::ofstream::binary);
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
 				tree.type = m_CurrentObject->GetType();
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
						SerializeNode(tree.GetRoot(), Context::Create(m_CurrentObject, m_CurrentObject->GetType()));
					}
				}
				else
				{
					SerializeNode(tree.GetRoot(), Context::Create(m_CurrentObject, m_CurrentObject->GetType()));
				}
				m_Trees.push_back(std::move(tree));
			}
			if (isText)
			{
				YamlWriter::Write(m_Trees, stream, true);
			}
			else
			{
				BinaryWriter::Write(m_Trees, stream);
			}
			stream.close();
		}
	}

	void EditorSerializer::Deserialize(const String& path)
	{
		std::ifstream stream(path.data(), std::ios::in | std::ofstream::binary);
		if (stream.is_open())
		{
			m_Trees.clear();
			switch (stream.peek())
			{
			case '%':
				YamlReader::Read(m_Trees, stream, true);
				break;
			case 'B':
				BinaryReader::Read(m_Trees, stream);
				break;
			}
			for (auto& tree : m_Trees)
			{
				Object* instance = nullptr;
				auto it = m_FileIdToObjectId.find(tree.fileId);
				if (it == m_FileIdToObjectId.end())
				{
					const ClassInfo* info = ClassDB::GetInfo(tree.type);
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
				AddDeserializedObject(instance->GetObjectId(), tree.fileId);
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
		FinalizeObjects();
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
				AssetDB::FinalizeObject(object, m_Guid, tree.fileId);
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
					instance->OnCreate();
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
					instance->OnCreate();
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
				Component* component = entity->GetComponent(i);
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
				AddObject(entity->GetComponent(i));
			}
		}
		Transform* transform = entity->GetTransform();
		for (auto& child : transform->GetChildren())
		{
			GatherChildrenPrefabs(child.Get()->GetEntity());
		}
	}
}
