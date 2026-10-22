#include "UndoDestroyComponentRecord.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Component.h"
#include "Editor\Prefabs\PrefabManager.h"
#include "Editor\Events\SceneEvents.h"

namespace Blueberry
{
	UndoDestroyComponentRecord::UndoDestroyComponentRecord(Component* component) : m_Entity(component->GetEntity()), m_Component(component)
	{
		m_Index = m_Entity->GetComponentIndex(component);
		m_Serializer.Serialize({ component });
		if (PrefabManager::IsPartOfPrefabInstance(component->GetEntity()))
		{
			PrefabManager::RemoveComponent(component);
		}
		m_Entity->RemoveComponent(component);
		Entity::Poll();
		SceneEvents::GetComponentDestroyed().Invoke();
	}

	void UndoDestroyComponentRecord::OnUndo()
	{
		m_Serializer.Deserialize();
		Entity* entity = m_Entity.Get();
		Component* component = m_Component.Get();
		if (PrefabManager::IsPartOfPrefabInstance(entity))
		{
			PrefabManager::AddComponent(entity, component);
		}
		entity->AddComponentAt(component, m_Index);
		Entity::Poll();
		SceneEvents::GetComponentCreated().Invoke();
	}

	void UndoDestroyComponentRecord::OnRedo()
	{
		Entity* entity = m_Entity.Get();
		Component* component = m_Component.Get();
		if (PrefabManager::IsPartOfPrefabInstance(entity))
		{
			PrefabManager::RemoveComponent(component);
		}
		m_Entity->RemoveComponent(component);
		Entity::Poll();
		SceneEvents::GetComponentDestroyed().Invoke();
	}
}