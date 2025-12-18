#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Variant.h"

namespace Blueberry
{
	class Object;

	class ObjectHelper
	{
	public:
		static void ReadValue(Object* object, const String& path, Variant& value);
		static void WriteValue(Object* object, const String& path, Variant& value);
	};
}