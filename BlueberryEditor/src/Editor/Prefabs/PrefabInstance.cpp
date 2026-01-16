#include "PrefabInstance.h"

#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Core\ObjectCloner.h"
#include "Editor\Prefabs\PrefabManager.h"
#include "Editor\Misc\ObjectHelper.h"

namespace Blueberry
{
	DATA_DEFINITION(PrefabModificationData)
	{
		DEFINE_FIELD(PrefabModificationData, m_Target, BindingType::ObjectPtr, FieldOptions().SetObjectType(Object::Type))
		DEFINE_FIELD(PrefabModificationData, m_Path, BindingType::String, {})
		DEFINE_FIELD(PrefabModificationData, m_Value, BindingType::Variant, {})
	}

	DATA_DEFINITION(PrefabAddedEntityData)
	{
		DEFINE_FIELD(PrefabAddedEntityData, m_Parent, BindingType::ObjectPtr, FieldOptions().SetObjectType(Transform::Type))
		DEFINE_FIELD(PrefabAddedEntityData, m_Entity, BindingType::ObjectPtr, FieldOptions().SetObjectType(Entity::Type))
		DEFINE_FIELD(PrefabAddedEntityData, m_Index, BindingType::Int, {})
	}

	OBJECT_DEFINITION(PrefabInstance, Object)
	{
		DEFINE_BASE_FIELDS(PrefabInstance, Object)
		DEFINE_FIELD(PrefabInstance, m_Prefab, BindingType::ObjectPtr, FieldOptions().SetObjectType(Entity::Type))
		DEFINE_FIELD(PrefabInstance, m_Parent, BindingType::ObjectPtr, FieldOptions().SetObjectType(Transform::Type))
		DEFINE_FIELD(PrefabInstance, m_Modifications, BindingType::DataList, FieldOptions().SetObjectType(PrefabModificationData::Type))
		DEFINE_FIELD(PrefabInstance, m_AddedEntities, BindingType::DataList, FieldOptions().SetObjectType(PrefabAddedEntityData::Type))
	}

	Object* PrefabModificationData::GetTarget()
	{
		return m_Target.Get();
	}

	void PrefabModificationData::SetTarget(Object* target)
	{
		m_Target = target;
	}

	const String& PrefabModificationData::GetPath()
	{
		return m_Path;
	}

	void PrefabModificationData::SetPath(const String& path)
	{
		m_Path = path;
	}

	Variant& PrefabModificationData::GetValue()
	{
		return m_Value;
	}

	void PrefabModificationData::SetValue(const Variant& value)
	{
		m_Value = value;
	}

	Transform* PrefabAddedEntityData::GetParent()
	{
		return m_Parent.Get();
	}

	void PrefabAddedEntityData::SetParent(Transform* parent)
	{
		m_Parent = parent;
	}

	Entity* PrefabAddedEntityData::GetEntity()
	{
		return m_Entity.Get();
	}

	void PrefabAddedEntityData::SetEntity(Entity* entity)
	{
		m_Entity = entity;
	}

	const int32_t& PrefabAddedEntityData::GetIndex()
	{
		return m_Index;
	}

	void PrefabAddedEntityData::SetIndex(const int32_t& index)
	{
		m_Index = index;
	}

	Entity* PrefabInstance::GetEntity()
	{
		return m_Entity.Get();
	}

	Object* PrefabInstance::GetCorrespondingObject(const FileId& fileId)
	{
		auto it = m_FileIdToObject.find(fileId);
		if (it != m_FileIdToObject.end())
		{
			return static_cast<Entity*>(ObjectDB::GetObject(it->second));
		}
		return nullptr;
	}

	void PrefabInstance::Update()
	{
		m_Parent = GetEntity()->GetTransform()->GetParent();

		for (auto& addedEntity : m_AddedEntities)
		{
			Transform* parent = addedEntity.GetParent();
			Transform* transform = addedEntity.GetEntity()->GetTransform();

			for (size_t i = 0; i < parent->GetChildrenCount(); ++i)
			{
				if (parent->GetChild(i) == transform)
				{
					addedEntity.SetIndex(i);
					break;
				}
			}
		}
	}

	void PrefabInstance::AddObjectMapping(Object* prefabObject, Object* instanceObject)
	{
		m_PrefabToInstanceMapping.insert_or_assign(prefabObject->GetObjectId(), instanceObject->GetObjectId());
	}

	void PrefabInstance::Resolve()
	{
		if (!m_Prefab.IsValid() || m_Prefab->GetState() != ObjectState::Default)
		{
			return;
		}
		bool isInitialization = m_Entity == nullptr;
		m_Entity = static_cast<Entity*>(ObjectCloner::Resolve(m_PrefabToInstanceMapping, m_Prefab.Get()));
		PrefabManager::s_RootToPrefabInstance.insert_or_assign(m_Entity->GetObjectId(), GetObjectId());
		for (auto& pair : m_PrefabToInstanceMapping)
		{
			Object* object = ObjectDB::GetObject(pair.first);
			if (object->IsClassType(Entity::Type))
			{
				Entity* entity = static_cast<Entity*>(object);
				if (PrefabManager::IsPrefabInstanceRoot(entity))
				{
					PrefabInstance* instance = PrefabManager::GetInstance(object);
					PrefabInstance* instanceClone = static_cast<PrefabInstance*>(ObjectCloner::Clone(instance));
					PrefabManager::s_RootToPrefabInstance.insert_or_assign(pair.second, instanceClone->GetObjectId());
				}
			}
			FileId fileId = ObjectDB::GetFileIdFromObject(object);	// Maybe instead manually get fileId of prefab object
			ObjectDB::AllocateIdToFileId(pair.second, fileId);
			PrefabManager::s_ObjectToPrefabInstance.insert_or_assign(pair.second, GetObjectId());
			PrefabManager::s_ObjectToPrefabObject.insert_or_assign(pair.second, pair.first);
			m_FileIdToObject.insert_or_assign(fileId, pair.second);
		}
		
		Transform* root = m_Entity->GetTransform();
		const ClassInfo* classInfo = ClassDB::GetInfo(Transform::Type);
		
		if (isInitialization)
		{
			classInfo->GetField("m_Parent")->Set(root, m_Parent);
			classInfo->GetField("m_Entity")->Set(root, m_Entity);
		}

		for (auto& pair : m_PrefabToInstanceMapping)
		{
			m_InstanceToPrefabMapping.insert_or_assign(pair.second, pair.first);
		}

		for (auto& modification : m_Modifications)
		{
			Object* target = modification.GetTarget();
			if (target != nullptr)
			{
				auto it = m_PrefabToInstanceMapping.find(target->GetObjectId());
				if (it != m_PrefabToInstanceMapping.end())
				{
					Object* object = ObjectDB::GetObject(it->second);
					String path = modification.GetPath();
					Variant& value = modification.GetValue();
					ObjectHelper::WriteValue(object, path, value);
				}
			}
		}
		
		for (auto& addedEntity : m_AddedEntities)
		{
			Transform* parent = addedEntity.GetParent();
			Entity* entity = addedEntity.GetEntity();
			if (parent != nullptr && entity != nullptr)
			{
				Transform* transform = entity->GetTransform();
				List<ObjectPtr<Transform>>* children = classInfo->GetField("m_Children")->Get<List<ObjectPtr<Transform>>>(root);
				int32_t index = addedEntity.GetIndex();
				if (index == -1)
				{
					children->push_back(transform);
				}
				else
				{
					children->insert(children->begin() + index, transform);
				}
			}
		}
	}

	void PrefabInstance::OnCreate()
	{
		if (m_Prefab.IsValid() && m_Entity == nullptr)
		{
			Resolve();
		}
	}

	void PrefabInstance::OnDestroy()
	{
		for (auto& pair : m_PrefabToInstanceMapping)
		{
			// This breaks nested root too
			// Need to rethink them
			PrefabManager::s_ObjectToPrefabInstance.erase(pair.second);
			PrefabManager::s_ObjectToPrefabObject.erase(pair.second);
		}

		if (m_Entity.IsValid())
		{
			PrefabManager::s_RootToPrefabInstance.erase(m_Entity->GetObjectId());
			Object::Destroy(m_Entity.Get());
		}
	}

	PrefabInstance* PrefabInstance::Create(Entity* prefab)
	{
		PrefabInstance* instance = Object::Create<PrefabInstance>();
		instance->m_Prefab = prefab;
		instance->OnCreate();
		return instance;
	}
}
