#include "EditorObjectManager.h"

#include "EditorSceneManager.h"
#include "Blueberry\Scene\Scene.h"

namespace Blueberry
{
	EntityCreateEvent EditorObjectManager::s_EntityCreated = {};
	EntityDestroyEvent EditorObjectManager::s_EntityDestroyed = {};

	Entity* EditorObjectManager::CreateEntity(const String& name)
	{
		Entity* entity = EditorSceneManager::GetScene()->CreateEntity(name);
		s_EntityCreated.Invoke();
		return entity;
	}

	void EditorObjectManager::DestroyEntity(Entity* entity)
	{
		EditorSceneManager::GetScene()->DestroyEntity(entity);
		s_EntityDestroyed.Invoke();
	}

	EntityCreateEvent& EditorObjectManager::GetEntityCreated()
	{
		return s_EntityCreated;
	}

	EntityDestroyEvent& EditorObjectManager::GetEntityDestroyed()
	{
		return s_EntityDestroyed;
	}
}
