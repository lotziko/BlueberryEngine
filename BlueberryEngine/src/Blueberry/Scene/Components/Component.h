#pragma once

#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class Entity;

	class Component : public Object
	{
		OBJECT_DECLARATION(Component)
	public:
		virtual ~Component() = default;

		Entity* GetEntity();

		static void BindProperties();

	private:
		Entity* m_Entity;

		friend class Entity;
	};
}