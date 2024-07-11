#include "bbpch.h"
#include "PrefabInstance.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Core\ObjectCloner.h"
#include "Editor\Prefabs\PrefabManager.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, PrefabInstance)

	Entity* PrefabInstance::GetEntity()
	{
		return m_Entity.Get();
	}

	void PrefabInstance::OnCreate()
	{
		// TODO handle changes too
		if (m_Prefab.IsValid())
		{
			m_Entity = (Entity*)ObjectCloner::Clone(m_Prefab.Get());
			PrefabManager::s_EntityToPrefabInstance.insert_or_assign(m_Entity->GetObjectId(), GetObjectId());
		}
	}

	void PrefabInstance::OnDestroy()
	{
		if (m_Entity.IsValid())
		{
			PrefabManager::s_EntityToPrefabInstance.erase(m_Entity->GetObjectId());
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

	void PrefabInstance::BindProperties()
	{
		BEGIN_OBJECT_BINDING(PrefabInstance)
		BIND_FIELD(FieldInfo(TO_STRING(m_Prefab), &PrefabInstance::m_Prefab, BindingType::ObjectPtr).SetObjectType(Entity::Type))
		END_OBJECT_BINDING()
	}
}
