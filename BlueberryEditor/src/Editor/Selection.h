#pragma once

#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class Selection
	{
	public:
		static Object* GetActiveObject();
		static void SetActiveObject(Object* object);

	private:
		static Object* s_ActiveObject;
	};
}