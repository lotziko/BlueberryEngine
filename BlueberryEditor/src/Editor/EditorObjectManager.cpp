#include "EditorObjectManager.h"

#include "EditorSceneManager.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Core\ObjectCloner.h"

namespace Blueberry
{
	EntityCreateEvent EditorObjectManager::s_EntityCreated = {};
	EntityDestroyEvent EditorObjectManager::s_EntityDestroyed = {};
	EntityUpdateEvent EditorObjectManager::s_EntityUpdated = {};

	Entity* EditorObjectManager::CreateEntity(const String& name)
	{
		Entity* entity = EditorSceneManager::GetScene()->CreateEntity(name);
		s_EntityCreated.Invoke();
		return entity;
	}

	Entity* EditorObjectManager::CloneEntity(Entity* entity)
	{
		Entity* newEntity = static_cast<Entity*>(ObjectCloner::Clone(entity));
		EditorSceneManager::GetScene()->AddEntity(newEntity);
		s_EntityCreated.Invoke();
		return newEntity;
	}

	void EditorObjectManager::AddEntity(Entity* entity)
	{
		EditorSceneManager::GetScene()->AddEntity(entity);
		s_EntityCreated.Invoke();
	}

	void EditorObjectManager::DestroyEntity(Entity* entity)
	{
		EditorSceneManager::GetScene()->DestroyEntity(entity);
		s_EntityDestroyed.Invoke();
	}

	void EditorObjectManager::AddComponent(Entity* entity, Component* component)
	{
		entity->AddComponent(component);
		s_EntityUpdated.Invoke();
	}

	void EditorObjectManager::RemoveComponent(Component* component)
	{
		component->GetEntity()->RemoveComponent(component);
		s_EntityUpdated.Invoke();
	}

	EntityCreateEvent& EditorObjectManager::GetEntityCreated()
	{
		return s_EntityCreated;
	}

	EntityDestroyEvent& EditorObjectManager::GetEntityDestroyed()
	{
		return s_EntityDestroyed;
	}

	EntityUpdateEvent& EditorObjectManager::GetEntityUpdated()
	{
		return s_EntityUpdated;
	}
}
