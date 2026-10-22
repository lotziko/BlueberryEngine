#pragma once

#include "Blueberry\Events\Event.h"

namespace Blueberry
{
	using SceneLoadEvent = Event<>;

	using EntityCreateEvent = Event<>;
	using EntityDestroyEvent = Event<>;
	using EntityUpdateEvent = Event<>;
	using ComponentCreateEvent = Event<>;
	using ComponentDestroyEvent = Event<>;
	using ObjectValueUpdateEvent = Event<>; // TODO Object ptr
	using HierarchyUpdateEvent = Event<>;

	class SceneEvents
	{
	public:
		static SceneLoadEvent& GetSceneLoaded();

		static EntityCreateEvent& GetEntityCreated();
		static EntityDestroyEvent& GetEntityDestroyed();
		static EntityUpdateEvent& GetEntityUpdated();
		static ComponentCreateEvent& GetComponentCreated();
		static ComponentDestroyEvent& GetComponentDestroyed();
		static ObjectValueUpdateEvent& GetObjectValueUpdated();
		static HierarchyUpdateEvent& GetHierarchyUpdated();

	private:
		static SceneLoadEvent s_SceneLoaded;

		static EntityCreateEvent s_EntityCreated;
		static EntityDestroyEvent s_EntityDestroyed;
		static ComponentCreateEvent s_ComponentCreated;
		static ComponentDestroyEvent s_ComponentDestroyed;
		static EntityUpdateEvent s_EntityUpdated;
		static ObjectValueUpdateEvent s_ObjectValueUpdated;
		static HierarchyUpdateEvent s_HierarchyUpdated;
	};
}