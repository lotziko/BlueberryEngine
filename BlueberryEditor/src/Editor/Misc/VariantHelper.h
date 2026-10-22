#pragma once

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Core\Variant.h"

namespace Blueberry
{
	class VariantHelper
	{
	public:
		static BindingType GetChildType(BindingType type);
		static void GetDefaultValue(BindingType type, Variant& value);
		static void ReadValue(BindingType type, void* ptr, Variant& value);
		static void WriteValue(BindingType type, void* ptr, Variant& value);
	};
}