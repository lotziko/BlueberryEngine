#pragma once

#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	class ObjectInspector
	{
	public:
		virtual void Draw(Object* object);
	private:
		void DrawField(Object* object, FieldInfo& info);
	};
}