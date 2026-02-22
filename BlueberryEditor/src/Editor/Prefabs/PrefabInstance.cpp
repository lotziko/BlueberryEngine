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
		DEFINE_FIELD(PrefabInstance, m_SourcePrefab, BindingType::ObjectPtr, FieldOptions().SetObjectType(PrefabInstance::Type))
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

	bool PrefabInstance::HasSource()
	{
		return m_SourcePrefab.IsValid();
	}

	PrefabInstance* PrefabInstance::GetSource()
	{
		return m_SourcePrefab.Get();
	}

	void PrefabInstance::OnCreate()
	{
		if (m_SourcePrefab.IsValid())
		{
			PrefabManager::InitializeHierarchy(this);
		}
		else
		{
			PrefabManager::InitializeContext(this);
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
			//Object::Destroy(m_Entity.Get());
		}
	}

	void PrefabInstance::UpdateIfNeeded()
	{
		PrefabInstance* sourcePrefab = m_SourcePrefab.Get();
		if (sourcePrefab != nullptr && sourcePrefab->m_UpdateCount > std::max(m_UpdateCount, 1u))
		{
			m_UpdateCount = sourcePrefab->m_UpdateCount;
			PrefabManager::InitializeHierarchy(this);
		}
	}

	void PrefabInstance::PrepareData()
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
					addedEntity.SetIndex(static_cast<int32_t>(i));
					break;
				}
			}
		}
	}

	void PrefabInstance::AddObjectMapping(Object* prefabObject, Object* instanceObject)
	{
		m_PrefabToInstanceMapping.insert_or_assign(prefabObject->GetObjectId(), instanceObject->GetObjectId());
	}
}
