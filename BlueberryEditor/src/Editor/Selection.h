#pragma once

#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Selection
	{
	public:
		static Object* GetActiveObject();
		static bool IsActiveObject(Object* object);
		static void AddActiveObject(Object* object);
		static void SetActiveObject(Object* object);

	private:
		static std::set<ObjectId> s_ActiveObjects;
	};
}