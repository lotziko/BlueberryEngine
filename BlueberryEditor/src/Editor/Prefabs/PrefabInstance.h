#pragma once

#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Entity;

	class PrefabInstance : public Object
	{
		OBJECT_DECLARATION(PrefabInstance)

		PrefabInstance() = default;
		virtual ~PrefabInstance() = default;

		Entity* GetEntity();

		virtual void OnCreate() final;
		virtual void OnDestroy() final;

		static PrefabInstance* Create(Entity* prefab);

		static void BindProperties();

	private:
		ObjectPtr<Entity> m_Prefab;
		ObjectPtr<Entity> m_Entity;

		friend class PrefabManager;
	};
}