#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class StringHelper
	{
	public:
		static void Split(const char* data, const char symbol, List<String>& result);
		static int32_t HasSubstring(const String& str1, const String& str2);
	};
}