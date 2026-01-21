#include "PrefabManager.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Serialization\Serializer.h"
#include "Editor\Prefabs\PrefabInstance.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Misc\ObjectHelper.h"

namespace Blueberry
{
	Dictionary<ObjectId, ObjectId> PrefabManager::s_RootToPrefabInstance = {};
	Dictionary<ObjectId, ObjectId> PrefabManager::s_ObjectToPrefabInstance = {};
	Dictionary<ObjectId, ObjectId> PrefabManager::s_ObjectToPrefabObject = {};

	bool PrefabManager::IsPrefabInstanceRoot(Entity* entity)
	{
		if (entity == nullptr)
		{
			return false;
		}
		return s_RootToPrefabInstance.count(entity->GetObjectId()) > 0;
	}

	bool PrefabManager::IsPartOfPrefabInstance(Object* object)
	{
		if (object == nullptr)
		{
			return false;
		}
		return s_ObjectToPrefabInstance.count(object->GetObjectId()) > 0;
	}

	PrefabInstance* PrefabManager::GetInstance(Object* object)
	{
		if (object == nullptr)
		{
			return nullptr;
		}
		auto it = s_ObjectToPrefabInstance.find(object->GetObjectId());
		if (it != s_ObjectToPrefabInstance.end())
		{
			return static_cast<PrefabInstance*>(ObjectDB::GetObject(it->second));
		}
		return nullptr;
	}

	PrefabInstance* PrefabManager::CreateInstance(Entity* prefabEntity)
	{
		if (prefabEntity == nullptr)
		{
			return nullptr;
		}
		// Check if entity is prefab
		if (!ObjectDB::HasGuid(prefabEntity))
		{
			return nullptr;
		}

		PrefabInstance* instance = PrefabInstance::Create(prefabEntity);
		return instance;
	}

	Object* PrefabManager::GetCorrespondingPrefabObject(Object* object)
	{
		if (object == nullptr)
		{
			return nullptr;
		}
		auto it = s_ObjectToPrefabObject.find(object->GetObjectId());
		if (it != s_ObjectToPrefabInstance.end())
		{
			return ObjectDB::GetObject(it->second);
		}
		return nullptr;
	}

	void PrefabManager::CreatePrefab(const String& path, Entity* entity)
	{
		if (entity == nullptr)
		{
			return;
		}
		String prefabName(entity->GetName());
		prefabName.append(".prefab");

		auto relativePath = std::filesystem::relative(path, Path::GetAssetsPath());
		relativePath.append(prefabName);

		AssetDB::CreateAsset(entity, relativePath.string().data());
		AssetDB::SaveAssets();
		AssetDB::Refresh();
	}

	void PrefabManager::UnpackPrefabInstance(Entity* entity)
	{
		if (entity == nullptr)
		{
			return;
		}
		auto it = s_RootToPrefabInstance.find(entity->GetObjectId());
		if (it != s_RootToPrefabInstance.end())
		{
			Object* object = ObjectDB::GetObject(it->second);
			PrefabInstance* instance = static_cast<PrefabInstance*>(object);
			instance->m_Entity = nullptr;
			instance->OnDestroy();
			Object::Destroy(instance);
			s_RootToPrefabInstance.erase(it->first);
		}
	}

	void PrefabManager::AddModification(Object* object, const String& path, Variant& value)
	{
		// TODO list resizing
		if (object == nullptr)
		{
			return;
		}
		PrefabInstance* instance = GetInstance(object);
		
		Object* target;
		auto it = instance->m_InstanceToPrefabMapping.find(object->GetObjectId());
		if (it != instance->m_InstanceToPrefabMapping.end())
		{
			target = ObjectDB::GetObject(it->second);
		}
		else
		{
			return;
		}

		for (auto& modification : instance->m_Modifications)
		{
			if (modification.GetTarget() == target && modification.GetPath() == path)
			{
				modification.SetValue(value);
				return;
			}
		}

		PrefabModificationData modification = {};
		modification.SetTarget(target);
		modification.SetPath(path);
		modification.SetValue(value);
		instance->m_Modifications.push_back(std::move(modification));
	}

	void PrefabManager::RemoveModification(Object* object, const String& path)
	{
		if (object == nullptr)
		{
			return;
		}
		PrefabInstance* instance = GetInstance(object);

		Object* target;
		auto it = instance->m_InstanceToPrefabMapping.find(object->GetObjectId());
		if (it != instance->m_InstanceToPrefabMapping.end())
		{
			target = ObjectDB::GetObject(it->second);
		}
		else
		{
			return;
		}

		for (auto it = instance->m_Modifications.begin(); it != instance->m_Modifications.end(); ++it)
		{
			if (it->GetTarget() == target && it->GetPath() == path)
			{
				instance->m_Modifications.erase(it);
				Object* prefabObject = ObjectDB::GetObject(instance->m_InstanceToPrefabMapping[object->GetObjectId()]);
				Variant value;
				ObjectHelper::ReadValue(prefabObject, path, value);
				ObjectHelper::WriteValue(object, path, value);
				break;
			}
		}
	}

	const bool PrefabManager::HasModification(Object* object, const String& path)
	{
		PrefabInstance* instance = GetInstance(object);

		Object* target;
		auto it = instance->m_InstanceToPrefabMapping.find(object->GetObjectId());
		if (it != instance->m_InstanceToPrefabMapping.end())
		{
			target = ObjectDB::GetObject(it->second);
		}
		else
		{
			return false;
		}

		for (auto it = instance->m_Modifications.begin(); it != instance->m_Modifications.end(); ++it)
		{
			if (it->GetTarget() == target && it->GetPath() == path)
			{
				return true;
			}
		}
		return false;
	}

	void PrefabManager::SetParent(Entity* entity, Transform* parent)
	{
		PrefabInstance* instance = GetInstance(parent);

		for (auto it = instance->m_AddedEntities.begin(); it != instance->m_AddedEntities.end(); ++it)
		{
			if (it->GetEntity() == entity)
			{
				it->SetParent(parent);
				return;
			}
		}

		PrefabAddedEntityData addedEntity = {};
		addedEntity.SetParent(parent);
		addedEntity.SetEntity(entity);
		instance->m_AddedEntities.push_back(addedEntity);
	}

	void PrefabManager::RemoveParent(Entity* entity)
	{
		Transform* parent = entity->GetTransform()->GetParent();
		PrefabInstance* instance = GetInstance(parent);

		for (auto it = instance->m_AddedEntities.begin(); it != instance->m_AddedEntities.end(); ++it)
		{
			if (it->GetEntity() == entity && it->GetParent() == parent)
			{
				instance->m_AddedEntities.erase(it);
				return;
			}
		}
	}

	bool PrefabManager::IsPrefabChild(Entity* entity)
	{
		return IsPartOfPrefabInstance(entity->GetTransform()->GetParent());
	}

	void PrefabManager::GatherScenePrefabs(Scene* scene, Serializer& serializer)
	{
		for (auto& rootEntity : scene->GetRootEntities())
		{
			GatherChildrenPrefabs(rootEntity.Get(), serializer);
		}
	}

	void PrefabManager::GatherChildrenPrefabs(Entity* entity, Serializer& serializer)
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
				serializer.AddObject(instance);
			}
			for (size_t i = 0; i < entity->GetComponentCount(); ++i)
			{
				Component* component = entity->GetComponent(i);
				if (!PrefabManager::IsPartOfPrefabInstance(component))
				{
					serializer.AddObject(component);
				}
			}
		}
		else
		{
			serializer.AddObject(entity);
			for (size_t i = 0; i < entity->GetComponentCount(); ++i)
			{
				serializer.AddObject(entity->GetComponent(i));
			}
		}
		Transform* transform = entity->GetTransform();
		for (auto& child : transform->GetChildren())
		{
			GatherChildrenPrefabs(child.Get()->GetEntity(), serializer);
		}
	}
}
