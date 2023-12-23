#pragma once

#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Selection
	{
	public:
		static Object* GetActiveObject();
		static void SetActiveObject(Object* object);

	private:
		static ObjectPtr<Object> s_ActiveObject;
	};
}