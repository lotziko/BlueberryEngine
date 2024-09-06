#pragma once

#include "Blueberry\Events\Event.h"

namespace Blueberry
{
	class Entity;

	using EntityCreateEvent = Event<>;
	using EntityDestroyEvent = Event<>;

	class EditorObjectManager
	{
	public:
		static Entity* CreateEntity(const std::string& name);
		static void DestroyEntity(Entity* entity);

		static EntityCreateEvent& GetEntityCreated();
		static EntityDestroyEvent& GetEntityDestroyed();

	private:
		static EntityCreateEvent s_EntityCreated;
		static EntityDestroyEvent s_EntityDestroyed;
	};
}