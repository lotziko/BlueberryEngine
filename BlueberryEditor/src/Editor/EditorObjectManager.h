#pragma once

#include "Blueberry\Events\Event.h"

namespace Blueberry
{
	class Entity;
	class Component;

	using EntityCreateEvent = Event<>;
	using EntityDestroyEvent = Event<>;
	using EntityUpdateEvent = Event<>;

	class EditorObjectManager
	{
	public:
		static Entity* CreateEntity(const String& name);
		static Entity* CloneEntity(Entity* entity);
		static void AddEntity(Entity* entity);
		static void DestroyEntity(Entity* entity);

		static void AddComponent(Entity* entity, Component* component);
		static void RemoveComponent(Component* component);

		static EntityCreateEvent& GetEntityCreated();
		static EntityDestroyEvent& GetEntityDestroyed();
		static EntityUpdateEvent& GetEntityUpdated();

	private:
		static EntityCreateEvent s_EntityCreated;
		static EntityDestroyEvent s_EntityDestroyed;
		static EntityUpdateEvent s_EntityUpdated;
	};
}