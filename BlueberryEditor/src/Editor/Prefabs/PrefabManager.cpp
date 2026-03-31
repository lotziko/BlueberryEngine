#include "PrefabManager.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Serialization\Serializer.h"
#include "Blueberry\Core\ObjectCloner.h"
#include "Editor\Prefabs\PrefabInstance.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Misc\ObjectHelper.h"
#include "Editor\Assets\AssetImporter.h"

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

	bool PrefabManager::IsPartOfPrefabInstance(const ObjectId& objectId)
	{
		bool isPart = s_ObjectToPrefabInstance.count(objectId) > 0;
		return isPart;
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

	PrefabInstance* PrefabManager::GetInstance(const ObjectId& objectId)
	{
		auto it = s_ObjectToPrefabInstance.find(objectId);
		if (it != s_ObjectToPrefabInstance.end())
		{
			return static_cast<PrefabInstance*>(ObjectDB::GetObject(it->second));
		}
		return nullptr;
	}

	PrefabInstance* PrefabManager::CreateInstance(PrefabInstance* source)
	{
		if (source == nullptr)
		{
			return nullptr;
		}
		PrefabInstance* instance = Object::Create<PrefabInstance>();
		instance->m_SourcePrefab = source;
		instance->Initialize();
		return instance;
	}

	PrefabInstance* PrefabManager::CloneInstance(PrefabInstance* source)
	{
		PrefabInstance* instance = Object::Create<PrefabInstance>();
		instance->m_SourcePrefab = source->m_SourcePrefab;
		instance->m_Modifications = source->m_Modifications;
		instance->m_UpdateCount = source->m_UpdateCount;
		// TODO added instances and components
		InitializeHierarchy(instance);
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
		if (IsPartOfPrefabInstance(entity))
		{
			return;
		}

		Entity* prefabEntity = static_cast<Entity*>(Object::Clone(entity));
		ClassDB::GetInfo(Transform::Type)->GetField("m_Parent")->Set(prefabEntity->GetTransform(), ObjectPtr<Transform>());
		String prefabName(prefabEntity->GetName());
		prefabName.append(".prefab");

		auto relativePath = std::filesystem::relative(path, Path::GetAssetsPath());
		relativePath.append(prefabName);

		AssetDB::CreateAsset(prefabEntity, relativePath.string().data());
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
		PrefabInstance* sourceInstance = instance->m_SourcePrefab.Get();
		//String guid = ObjectDB::GetGuidFromObject().ToString();
		Object* prefabObject = ObjectDB::GetObject(instance->m_InstanceToPrefabMapping[object->GetObjectId()]);
		for (auto& modification : instance->m_Modifications)
		{
			if (modification.GetTarget() == prefabObject && modification.GetPath() == path)
			{
				modification.SetValue(value);
				return;
			}
		}

		PrefabModificationData modification = {};
		modification.SetTarget(prefabObject);
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
		Object* prefabObject = ObjectDB::GetObject(instance->m_InstanceToPrefabMapping[object->GetObjectId()]);
		for (auto it = instance->m_Modifications.begin(); it != instance->m_Modifications.end(); ++it)
		{
			if (it->GetTarget() == prefabObject && it->GetPath() == path)
			{
				instance->m_Modifications.erase(it);
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
		if (instance == nullptr)
		{
			return false;
		}
		Object* prefabObject = ObjectDB::GetObject(instance->m_InstanceToPrefabMapping[object->GetObjectId()]);
		for (auto it = instance->m_Modifications.begin(); it != instance->m_Modifications.end(); ++it)
		{
			if (it->GetTarget() == prefabObject && it->GetPath() == path)
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

	bool PrefabManager::IsOverridable(Object* object)
	{
		PrefabInstance* instance = GetInstance(object);
		if (instance != nullptr && instance->m_SourcePrefab.IsValid())
		{
			return true;
		}
		return false;
	}

	void PrefabManager::InitializeHierarchy(PrefabInstance* instance)
	{
		ObjectId instanceObjectId = instance->GetObjectId();
		PrefabInstance* sourcePrefab = instance->m_SourcePrefab.Get();
		if (sourcePrefab != nullptr)
		{
			Guid guid = ObjectDB::GetGuidFromObject(sourcePrefab);
			AssetImporter* importer = AssetDB::GetImporter(guid);
			Entity* assetEntity = static_cast<Entity*>(ObjectDB::GetObjectFromGuid(guid, importer->GetMainObject()));
			if (assetEntity != nullptr)
			{
				List<ObjectId> removed;
				Entity* entity = static_cast<Entity*>(ObjectCloner::Resolve(instance->m_PrefabToInstanceMapping, removed, assetEntity));
				Transform* transform = entity->GetTransform();
				instance->m_Entity = entity;
				s_RootToPrefabInstance.insert_or_assign(entity->GetObjectId(), instanceObjectId);
				for (auto& pair : instance->m_PrefabToInstanceMapping)
				{
					FileId fileId = ObjectDB::GetFileIdFromObjectId(pair.first);
					Object* object = ObjectDB::GetObject(pair.first);
					s_ObjectToPrefabInstance.insert_or_assign(pair.second, instanceObjectId);
					s_ObjectToPrefabObject.insert_or_assign(pair.second, pair.first);
					instance->m_FileIdToObject.insert_or_assign(fileId, pair.second);
				}
				for (ObjectId id : removed)
				{
					Object* object = ObjectDB::GetObject(id);
					if (object != nullptr)
					{
						if (object->IsClassType(Entity::Type))
						{
							Entity* entity = static_cast<Entity*>(object);
							Scene* scene = entity->GetScene();
							if (scene != nullptr)
							{
								scene->DestroyEntity(entity);
							}
						}
						else
						{
							Object::Destroy(object);
						}
					}
				}
				const ClassInfo* classInfo = ClassDB::GetInfo(Transform::Type);
				classInfo->GetField("m_Parent")->Set(transform, instance->m_Parent);
				classInfo->GetField("m_Entity")->Set(transform, instance->m_Entity);

				for (auto& pair : instance->m_PrefabToInstanceMapping)
				{
					instance->m_InstanceToPrefabMapping.insert_or_assign(pair.second, pair.first);
				}

				for (auto& modification : instance->m_Modifications)
				{
					Object* target = modification.GetTarget();
					if (target != nullptr)
					{
						auto it = instance->m_PrefabToInstanceMapping.find(target->GetObjectId());
						if (it != instance->m_PrefabToInstanceMapping.end())
						{
							Object* object = ObjectDB::GetObject(it->second);
							String path = modification.GetPath();
							Variant& value = modification.GetValue();
							ObjectHelper::WriteValue(object, path, value);
						}
					}
				}
			}
			instance->m_UpdateCount = sourcePrefab->m_UpdateCount;
		}
	}

	void PrefabManager::InitializeContext(PrefabInstance* instance)
	{
		if (!instance->m_SourcePrefab.IsValid())
		{
			ObjectId instanceObjectId = instance->GetObjectId();
			Guid guid = ObjectDB::GetGuidFromObject(instance);
			AssetImporter* importer = AssetDB::GetImporter(guid);
			Entity* entity = static_cast<Entity*>(ObjectDB::GetObjectFromGuid(guid, importer->GetMainObject()));
			if (entity == nullptr)
			{
				BB_ERROR("Can't initialize prefab context.");
				return;
			}
			instance->m_Entity = entity;
			s_RootToPrefabInstance.insert_or_assign(entity->GetObjectId(), instanceObjectId);
			InitializeChildContext(instance, guid, instance->m_Entity->GetTransform());
			++instance->m_UpdateCount;
		}
	}

	void PrefabManager::InitializeChildContext(PrefabInstance* instance, Guid guid, Transform* child)
	{
		if (child == nullptr)
		{
			BB_ERROR("Can't initialize prefab");
			return;
		}
		PrefabInstance* childInstance = GetInstance(child);
		Entity* entity = child->GetEntity();
		if (entity == nullptr)
		{
			BB_ERROR("Can't initialize prefab");
			return;
		}
		if (childInstance != nullptr && childInstance != instance)
		{
			FileId childInstanceFileId = ObjectDB::GetFileIdFromObject(childInstance);
			FileId entityFileId = ObjectDB::GetFileIdFromObjectId(childInstance->m_InstanceToPrefabMapping[entity->GetObjectId()]);
			if (entityFileId != 0)
			{
				ObjectDB::AllocateIdToGuid(entity, guid, entityFileId ^ childInstanceFileId);
				for (size_t i = 0; i < entity->GetComponentCount(); ++i)
				{
					Component* component = entity->GetComponentAt(i);
					FileId componentFileId = ObjectDB::GetFileIdFromObject(component);
					if (componentFileId != 0)
					{
						ObjectDB::AllocateIdToGuid(component, guid, componentFileId ^ childInstanceFileId);
					}
				}
			}
		}
		else
		{
			ObjectId rootInstanceObjectId = instance->GetObjectId();
			s_ObjectToPrefabInstance.insert_or_assign(entity->GetObjectId(), rootInstanceObjectId);
			for (size_t i = 0; i < entity->GetComponentCount(); ++i)
			{
				s_ObjectToPrefabInstance.insert_or_assign(entity->GetComponentAt(i)->GetObjectId(), rootInstanceObjectId);
			}
		}
		for (auto& child : child->GetChildren())
		{
			InitializeChildContext(instance, guid, child.Get());
		}
	}
}
