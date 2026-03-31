#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Component;
	class Entity;

	class SceneEvents
	{
	public:
		static void Poll();

	private:
		static List<ObjectPtr<Component>> s_CreatedComponents;
		static List<ObjectPtr<Component>> s_EnabledComponents;
		static List<ObjectPtr<Component>> s_DisabledComponents;
		static List<ObjectPtr<Component>> s_DestroyedComponents;
		static List<ObjectPtr<Entity>> s_DestroyedEntities;

		friend class Entity;
		friend class Scene;
	};
}