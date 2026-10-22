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
		ObjectMapping sourceToClone;
		HashSet<ObjectId> removed;

		Entity* instanceEntity = static_cast<Entity*>(ObjectCloner::Resolve(sourceToClone, removed, source->GetEntity()));
		PrefabInstance* instance = Object::Create<PrefabInstance>();
		instance->m_SourcePrefab = source->m_SourcePrefab;
		instance->m_Entity = instanceEntity;
		instance->m_Modifications = source->m_Modifications;
		instance->m_UpdateCount = source->m_UpdateCount;

		// Only prefab-backed objects belong in this mapping.
		for (auto& pair : source->m_PrefabToInstanceMapping)
		{
			auto it = sourceToClone.find(pair.second);
			if (it == sourceToClone.end())
			{
				continue;
			}

			instance->m_PrefabToInstanceMapping.insert_or_assign(pair.first, it->second);
		}

		auto findClone = [&](Object* object) -> Object*
		{
			if (object == nullptr)
			{
				return nullptr;
			}

			auto it = sourceToClone.find(object->GetObjectId());
			return it != sourceToClone.end() ? ObjectDB::GetObject(it->second) : nullptr;
		};

		for (auto& addedEntity : source->m_AddedEntities)
		{
			Entity* entity = addedEntity.GetEntity();
			Entity* clonedEntity = static_cast<Entity*>(findClone(entity));
			Transform* clonedParent = static_cast<Transform*>(findClone(addedEntity.GetParent()));

			if (clonedEntity == nullptr || clonedParent == nullptr)
			{
				continue;
			}

			PrefabAddedEntityData data = {};
			data.SetEntity(clonedEntity);
			data.SetParent(clonedParent);
			data.SetIndex(static_cast<int32_t>(entity->GetTransform()->GetSiblingIndex()));
			instance->m_AddedEntities.push_back(data);
		}

		for (auto& addedComponent : source->m_AddedComponents)
		{
			Entity* entity = static_cast<Entity*>(findClone(addedComponent.GetEntity()));
			Component* component = static_cast<Component*>(findClone(addedComponent.GetComponent()));

			if (entity == nullptr || component == nullptr)
			{
				continue;
			}

			PrefabAddedComponentData data = {};
			data.SetEntity(entity);
			data.SetComponent(component);
			instance->m_AddedComponents.push_back(data);
		}
		
		for (auto& modification : instance->m_Modifications)
		{
			auto* reference = std::get_if<ObjectPtr<Object>>(&modification.GetValue());
			if (reference != nullptr)
			{
				Object* clone = findClone(reference->Get());
				if (clone != nullptr)
				{
					*reference = clone;
				}
			}
		}

		ObjectId instanceObjectId = instance->GetObjectId();
		s_RootToPrefabInstance.insert_or_assign(instanceEntity->GetObjectId(), instanceObjectId);
		for (auto& pair : instance->m_PrefabToInstanceMapping)
		{
			FileId fileId = ObjectDB::GetFileIdFromObjectId(pair.first);
			s_ObjectToPrefabInstance.insert_or_assign(pair.second, instanceObjectId);
			s_ObjectToPrefabObject.insert_or_assign(pair.second, pair.first);
			instance->m_FileIdToObject.insert_or_assign(fileId, pair.second);
		}

		Transform* instanceTransform = instanceEntity->GetTransform();
		const ClassInfo* classInfo = ClassDB::GetInfo(Transform::Type);
		classInfo->GetField("m_Parent")->Set(instanceTransform, instance->m_Parent);

		for (auto& pair : instance->m_PrefabToInstanceMapping)
		{
			instance->m_InstanceToPrefabMapping.insert_or_assign(pair.second, pair.first);
		}

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

		Entity* prefabEntity = static_cast<Entity*>(ObjectCloner::Clone(entity));
		ClassDB::GetInfo(Transform::Type)->GetField("m_Parent")->Set(prefabEntity->GetTransform(), ObjectPtr<Transform>());
		String prefabName(prefabEntity->GetName());
		prefabName.append(".prefab");

		auto relativePath = std::filesystem::relative(path, Path::GetAssetsPath());
		relativePath.append(prefabName);

		AssetDB::CreateAsset(prefabEntity, StringHelper::ToString(relativePath));
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

	void PrefabManager::AddModification(Object* object, const String& path, const Variant& value)
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
				auto* objectRef = std::get_if<ObjectPtr<Object>>(&value);
				if (objectRef != nullptr && objectRef->IsValid())
				{
					auto mappingIt = instance->m_PrefabToInstanceMapping.find(objectRef->Get()->GetObjectId());
					if (mappingIt != instance->m_PrefabToInstanceMapping.end())
					{
						*objectRef = ObjectDB::GetObject(mappingIt->second);
					}
				}
				ObjectHelper::WriteValue(object, path, value);
				break;
			}
		}
	}

	bool PrefabManager::HasModification(Object* object, const String& path)
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

	void PrefabManager::AddComponent(Entity* entity, Component* component)
	{
		PrefabInstance* instance = GetInstance(entity);
		if (instance != nullptr && instance->HasSource())
		{
			PrefabAddedComponentData addedComponent = {};
			addedComponent.SetEntity(entity);
			addedComponent.SetComponent(component);
			instance->m_AddedComponents.push_back(addedComponent);
		}
	}

	void PrefabManager::RemoveComponent(Component* component)
	{
		Entity* entity = component->GetEntity();
		PrefabInstance* instance = GetInstance(entity);

		for (auto it = instance->m_AddedComponents.begin(); it != instance->m_AddedComponents.end(); ++it)
		{
			if (it->GetEntity() == entity && it->GetComponent() == component)
			{
				instance->m_AddedComponents.erase(it);
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
				HashSet<ObjectId> removed;
				Entity* instanceEntity = static_cast<Entity*>(ObjectCloner::Resolve(instance->m_PrefabToInstanceMapping, removed, assetEntity));
				Transform* instanceTransform = instanceEntity->GetTransform();
				instance->m_Entity = instanceEntity;
				s_RootToPrefabInstance.insert_or_assign(instanceEntity->GetObjectId(), instanceObjectId);
				for (auto& pair : instance->m_PrefabToInstanceMapping)
				{
					FileId fileId = ObjectDB::GetFileIdFromObjectId(pair.first);
					s_ObjectToPrefabInstance.insert_or_assign(pair.second, instanceObjectId);
					s_ObjectToPrefabObject.insert_or_assign(pair.second, pair.first);
					instance->m_FileIdToObject.insert_or_assign(fileId, pair.second);
				}
				
				// Remove mappings/objects
				for (ObjectId id : removed)
				{
					instance->m_InstanceToPrefabMapping.erase(id);
					auto it = s_ObjectToPrefabInstance.find(id);
					if (it != s_ObjectToPrefabInstance.end() && it->second == instanceObjectId)
					{
						s_ObjectToPrefabInstance.erase(it);
						s_ObjectToPrefabObject.erase(id);
					}
					for (auto& pair : instance->m_FileIdToObject)
					{
						if (pair.second == id)
						{
							instance->m_FileIdToObject.erase(pair.first);
							break;
						}
					}

					Object* object = ObjectDB::GetObject(id);
					if (object != nullptr)
					{
						if (object->IsClassType(Entity::Type))
						{
							// Destroy PrefabInstance of erased root
							auto rootIt = s_RootToPrefabInstance.find(id);
							if (rootIt != s_RootToPrefabInstance.end())
							{
								const ObjectId prefabInstanceId = rootIt->second;
								s_RootToPrefabInstance.erase(rootIt);
								if (prefabInstanceId != instanceObjectId)
								{
									PrefabInstance* nestedInstance = static_cast<PrefabInstance*>(ObjectDB::GetObject(prefabInstanceId));
									if (nestedInstance != nullptr)
									{
										Entity* nestedRoot = nestedInstance->GetEntity();
										if (nestedRoot == nullptr || nestedRoot->GetObjectId() == id)
										{
											Object::Destroy(nestedInstance);
										}
									}
								}
							}

							Entity* entity = static_cast<Entity*>(object);
							Scene* scene = entity->GetScene();
							if (scene != nullptr)
							{
								scene->DestroyEntity(entity);
							}
						}
						else if (object->IsClassType(Component::Type))
						{
							Component* component = static_cast<Component*>(object);
							component->GetEntity()->RemoveComponent(component);
						}
						else
						{
							Object::Destroy(object);
						}
					}
				}
				const ClassInfo* classInfo = ClassDB::GetInfo(Transform::Type);
				classInfo->GetField("m_Parent")->Set(instanceTransform, instance->m_Parent);
				classInfo->GetField("m_Entity")->Set(instanceTransform, instance->m_Entity);

				for (auto& pair : instance->m_PrefabToInstanceMapping)
				{
					instance->m_InstanceToPrefabMapping.insert_or_assign(pair.second, pair.first);
				}

				// Apply components
				for (auto& addedComponent : instance->m_AddedComponents)
				{
					Entity* target = addedComponent.GetEntity();
					Component* component = addedComponent.GetComponent();

					if (target == nullptr || component == nullptr)
					{
						continue;
					}

					if (removed.count(target->GetObjectId()) > 0)
					{
						continue;
					}

					bool attached = false;
					for (size_t i = 0; i < target->GetComponentCount(); ++i)
					{
						if (target->GetComponentAt(i) == component)
						{
							attached = true;
							break;
						}
					}

					if (!attached)
					{
						target->AddComponent(component);
					}
				}

				// Apply modifications
				HashSet<Object*> modified;
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
							modified.insert(object);
						}
					}
				}
				for (Object* object : modified)
				{
					object->OnPropertiesChanged();
				}

				// Apply entities
				if (!instance->m_AddedEntities.empty())
				{
					List<PrefabAddedEntityData> addedEntities = instance->m_AddedEntities;
					std::sort(addedEntities.begin(), addedEntities.end(), [](PrefabAddedEntityData& a, PrefabAddedEntityData& b) { return a.GetIndex() < b.GetIndex(); });
					for (auto& addedEntity : addedEntities)
					{
						Entity* entity = addedEntity.GetEntity();
						Transform* parent = addedEntity.GetParent();

						if (entity == nullptr || parent == nullptr || removed.count(entity->GetObjectId()) > 0 || removed.count(parent->GetObjectId()) > 0)
						{
							continue;
						}

						Transform* transform = entity->GetTransform();
						if (transform == nullptr)
						{
							continue;
						}
						classInfo->GetField("m_Parent")->Set(transform, ObjectPtr<Transform>());
						transform->SetParent(parent, false);
						const size_t index = static_cast<size_t>(std::max(addedEntity.GetIndex(), 0));
						transform->SetSiblingIndex(std::min(index, parent->GetChildrenCount() - 1));

						Scene* scene = parent->GetScene();
						if (scene != nullptr)
						{
							scene->AddEntity(entity);
						}
					}
				}
				instanceEntity->UpdateHierarchy();
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
			if (!guid.IsValid())
			{
				BB_ERROR("Prefab is invalid.");
				return;
			}
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
