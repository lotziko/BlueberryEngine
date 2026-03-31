#include "Blueberry\Tools\StringConverter.h"

#include <codecvt>

namespace Blueberry
{
	WString StringConverter::StringToWide(const String& str)
	{
		WString wide_string(str.begin(), str.end());
		return wide_string;
	}

	String StringConverter::WideToString(const WString& wstr)
	{
		static std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
		return String(converter.to_bytes(wstr.c_str()));
	}
}