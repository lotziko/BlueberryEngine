#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class StringConverter
	{
	public:
		static WString StringToWide(const String& str);
		static String WideToString(const WString& wstr);
	};
}