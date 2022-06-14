#pragma once

#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class Selection
	{
	public:
		static Object* GetActiveObject() { return s_ActiveObject; }
		static void SetActiveObject(Object* object) { s_ActiveObject = object; }

	private:
		static Object* s_ActiveObject;
	};
}