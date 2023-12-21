#pragma once

#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\WeakObjectPtr.h"

namespace Blueberry
{
	class Selection
	{
	public:
		static Object* GetActiveObject();
		static void SetActiveObject(Object* object);

	private:
		static WeakObjectPtr<Object> s_ActiveObject;
	};
}