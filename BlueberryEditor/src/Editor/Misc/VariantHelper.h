#pragma once

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Core\Variant.h"

namespace Blueberry
{
	class VariantHelper
	{
	public:
		static BindingType GetChildType(const BindingType& type);
		static void GetDefaultValue(const BindingType& type, Variant& value);
		static void ReadValue(const BindingType& type, void* ptr, Variant& value);
		static void WriteValue(const BindingType& type, void* ptr, Variant& value);
	};
}