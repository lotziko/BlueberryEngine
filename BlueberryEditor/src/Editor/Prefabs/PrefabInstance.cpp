#include "PrefabInstance.h"

#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Core\ObjectCloner.h"
#include "Editor\Prefabs\PrefabManager.h"

namespace Blueberry
{
	OBJECT_DEFINITION(PrefabInstance, Object)
	{
		DEFINE_BASE_FIELDS(PrefabInstance, Object)
		DEFINE_FIELD(PrefabInstance, m_Prefab, BindingType::ObjectPtr, FieldOptions().SetObjectType(Entity::Type))
	}

	Entity* PrefabInstance::GetEntity()
	{
		return m_Entity.Get();
	}

	void PrefabInstance::OnCreate()
	{
		// TODO handle changes too
		if (m_Prefab.IsValid())
		{
			m_Entity = static_cast<Entity*>(ObjectCloner::Clone(m_Prefab.Get()));
			PrefabManager::s_EntityToPrefabInstance.insert_or_assign(m_Entity->GetObjectId(), GetObjectId());
			AddPrefabEntities(m_Entity.Get());
		}
	}

	void PrefabInstance::OnDestroy()
	{
		if (m_Entity.IsValid())
		{
			PrefabManager::s_EntityToPrefabInstance.erase(m_Entity->GetObjectId());
			RemovePrefabEntities(m_Entity.Get());
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

	void PrefabInstance::AddPrefabEntities(Entity* entity)
	{
		PrefabManager::s_PrefabEntities.insert(entity->GetObjectId());
		for (auto& child : entity->GetTransform()->GetChildren())
		{
			AddPrefabEntities(child->GetEntity());
		}
	}

	void PrefabInstance::RemovePrefabEntities(Entity* entity)
	{
		PrefabManager::s_PrefabEntities.erase(entity->GetObjectId());
		for (auto& child : entity->GetTransform()->GetChildren())
		{
			RemovePrefabEntities(child->GetEntity());
		}
	}
}
