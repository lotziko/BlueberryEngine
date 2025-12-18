#include "YamlSceneSerializer.h"

#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Editor\Prefabs\PrefabManager.h"
#include "Editor\Prefabs\PrefabInstance.h"
#include "Editor\Serialization\YamlHelper.h"
#include "Editor\Serialization\YamlSerializers.h"

namespace Blueberry
{
	void YamlSceneSerializer::AddSceneObjects(Scene* scene)
	{
		for (auto& rootEntity : scene->GetRootEntities())
		{
			GatherSceneObjects(rootEntity.Get());
		}
	}

	void YamlSceneSerializer::Serialize(const String& path)
	{
		m_AssetGuid = ObjectDB::GetGuidFromObject(m_ObjectsToSerialize[0]);
		ryml::Tree tree;
		ryml::NodeRef root = tree.rootref();
		root |= ryml::MAP;

		Object* object;
		while ((object = GetNextObjectToSerialize()) != nullptr)
		{
			if (PrefabManager::IsPartOfPrefabInstance(object))
			{
				PrefabInstance* instance = PrefabManager::GetInstance(object);
				ryml::NodeRef objectNode = root.append_child() << ryml::key(object->GetTypeName());
				objectNode |= ryml::MAP;
				FileId id = GetFileId(object);
				objectNode.set_key_tag(ryml::to_csubstr(*(m_Tags.push_back(std::to_string(id).append("!reference")))));
				objectNode["m_PrefabInstance"] << GetPtrData(instance);
				objectNode["m_CorrespondingPrefabObject"] << GetPtrData(PrefabManager::GetCorrespondingPrefabObject(object));
			}
			else
			{
				FileId id = GetFileId(object);
				ryml::NodeRef objectNode = root.append_child() << ryml::key(object->GetTypeName());
				objectNode.set_key_tag(ryml::to_csubstr(*(m_Tags.push_back(std::to_string(id)))));
				objectNode |= ryml::MAP;
				SerializeNode(objectNode, Context::Create(object, object->GetType()));
			}
		}
		YamlHelper::Save(tree, path);
	}

	void YamlSceneSerializer::Deserialize(const String& path)
	{
		ryml::Tree tree;
		YamlHelper::Load(tree, path);
		ryml::NodeRef root = tree.rootref();
		List<std::pair<int, Object*>> deserializedReferences = {};
		List<std::pair<int, Object*>> deserializedNodes = {};
		List<PrefabInstance*> prefabInstances = {};

		for (size_t i = 0; i < root.num_children(); i++)
		{
			ryml::ConstNodeRef node = root[i];
			FileId fileId;
			if (node.has_key_tag())
			{
				bool isReference = false;
				ryml::csubstr tag = node.key_tag().trim('!');
				size_t referencePos = tag.find("reference");
				if (referencePos != ryml::npos)
				{
					isReference = true;
					tag = tag.sub(0, referencePos);
				}

				if (ryml::from_chars(tag.trim("!"), &fileId))
				{
					Object* instance;
					auto it = m_FileIdToObject.find(fileId);
					if (it == m_FileIdToObject.end())
					{
						ryml::csubstr key = node.key();
						String typeName(key.str, key.size());
						const ClassInfo* info = ClassDB::GetInfo(TO_OBJECT_TYPE(typeName));
						if (info == nullptr)
						{
							BB_ERROR("Class not exists.");
							continue;
						}
						instance = info->createInstance();
						AddDeserializedObject(instance, fileId);
					}
					else
					{
						instance = it->second;
						AddDeserializedObject(instance, fileId);
					}

					if (isReference)
					{
						deserializedReferences.push_back({ i, instance });
					}
					else
					{
						deserializedNodes.push_back({ i, instance });
						if (instance->IsClassType(PrefabInstance::Type))
						{
							prefabInstances.push_back(static_cast<PrefabInstance*>(instance));
						}
					}
				}
			}
		}

		for (auto& pair : deserializedNodes)
		{
			Object* object = pair.second;
			DeserializeNode(root[pair.first], Context::Create(object, object->GetType()));	
		}

		for (auto& pair : deserializedReferences)
		{
			ryml::ConstNodeRef node = root[pair.first];
			ObjectPtrData prefabInstanceData = {};
			ObjectPtrData correspondingPrefabObjectData = {};
			node["m_PrefabInstance"] >> prefabInstanceData;
			node["m_CorrespondingPrefabObject"] >> correspondingPrefabObjectData;
			PrefabInstance* prefabInstance = static_cast<PrefabInstance*>(GetPtrObject(prefabInstanceData));
			Object* correspondingPrefabObject = GetPtrObject(correspondingPrefabObjectData);
			prefabInstance->AddObjectMapping(correspondingPrefabObject, pair.second);
		}

		for (PrefabInstance* instance : prefabInstances)
		{
			instance->Resolve();
		}
	}

	void YamlSceneSerializer::GatherSceneObjects(Entity* entity)
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
				instance->Update();
				AddObject(instance);
			}
			return;
		}
		AddObject(entity);
		for (size_t i = 0; i < entity->GetComponentCount(); ++i)
		{
			AddObject(entity->GetComponent(i));
		}
		Transform* transform = entity->GetTransform();
		for (auto& child : transform->GetChildren())
		{
			GatherSceneObjects(child.Get()->GetEntity());
		}
	}
}
