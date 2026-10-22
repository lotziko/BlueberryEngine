#include "SceneEvents.h"

namespace Blueberry
{
	SceneLoadEvent SceneEvents::s_SceneLoaded = {};

	EntityCreateEvent SceneEvents::s_EntityCreated = {};
	EntityDestroyEvent SceneEvents::s_EntityDestroyed = {};
	EntityUpdateEvent SceneEvents::s_EntityUpdated = {};
	ComponentCreateEvent SceneEvents::s_ComponentCreated = {};
	ComponentDestroyEvent SceneEvents::s_ComponentDestroyed = {};
	ObjectValueUpdateEvent SceneEvents::s_ObjectValueUpdated = {};
	HierarchyUpdateEvent SceneEvents::s_HierarchyUpdated = {};

	SceneLoadEvent& SceneEvents::GetSceneLoaded()
	{
		return s_SceneLoaded;
	}

	EntityCreateEvent& SceneEvents::GetEntityCreated()
	{
		return s_EntityCreated;
	}

	EntityDestroyEvent& SceneEvents::GetEntityDestroyed()
	{
		return s_EntityDestroyed;
	}

	EntityUpdateEvent& SceneEvents::GetEntityUpdated()
	{
		return s_EntityUpdated;
	}

	ComponentCreateEvent& SceneEvents::GetComponentCreated()
	{
		return s_ComponentCreated;
	}

	ComponentDestroyEvent& SceneEvents::GetComponentDestroyed()
	{
		return s_ComponentDestroyed;
	}

	ObjectValueUpdateEvent& SceneEvents::GetObjectValueUpdated()
	{
		return s_ObjectValueUpdated;
	}

	HierarchyUpdateEvent& SceneEvents::GetHierarchyUpdated()
	{
		return s_HierarchyUpdated;
	}
}