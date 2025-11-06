#include "Blueberry\Tools\StringConverter.h"

namespace Blueberry
{
	WString StringConverter::StringToWide(String str)
	{
		WString wide_string(str.begin(), str.end());
		return wide_string;
	}
}