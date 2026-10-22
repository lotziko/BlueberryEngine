#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Entity;
	class Object;

	class EntityHelper
	{
	public:
		static Entity* Create(const String& name);
		static void GatherObjects(Entity* entity, List<Object*>& objects);
	};
}