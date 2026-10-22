#include "UndoAddComponentRecord.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Component.h"
#include "Editor\Prefabs\PrefabManager.h"
#include "Editor\Events\SceneEvents.h"

namespace Blueberry
{
	UndoAddComponentRecord::UndoAddComponentRecord(Entity* entity, Component* component) : m_Entity(entity), m_Component(component)
	{
		if (PrefabManager::IsPartOfPrefabInstance(entity))
		{
			PrefabManager::AddComponent(entity, component);
		}
		entity->AddComponent(component);
		Entity::Poll();
		SceneEvents::GetComponentCreated().Invoke();
	}

	void UndoAddComponentRecord::OnUndo()
	{
		Entity* entity = m_Entity.Get();
		Component* component = m_Component.Get();
		if (m_Serializer.IsEmpty())
		{
			m_Serializer.Serialize({ component });
		}
		if (PrefabManager::IsPartOfPrefabInstance(entity))
		{
			PrefabManager::RemoveComponent(component);
		}
		entity->RemoveComponent(component);
		Entity::Poll();
		SceneEvents::GetComponentDestroyed().Invoke();
	}

	void UndoAddComponentRecord::OnRedo()
	{
		m_Serializer.Deserialize();
		Entity* entity = m_Entity.Get();
		Component* component = m_Component.Get();
		if (PrefabManager::IsPartOfPrefabInstance(entity))
		{
			PrefabManager::AddComponent(entity, component);
		}
		entity->AddComponent(component);
		Entity::Poll();
		SceneEvents::GetComponentCreated().Invoke();
	}
}