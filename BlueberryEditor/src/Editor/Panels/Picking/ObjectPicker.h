#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Object;

	class ObjectPicker
	{
	public:
		static void Open(Object** object, const TypeId& type);
		static bool GetResult(Object** object);

	private:
		static bool DrawNone(const bool& selected);
		static bool DrawObject(Object* object, const bool& selected);
	};
}