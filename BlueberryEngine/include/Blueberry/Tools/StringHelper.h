#pragma once

#include "Blueberry\Core\Base.h"

namespace std::filesystem
{
	class path;
}

namespace Blueberry
{
	class StringHelper
	{
	public:
		static void Replace(String& str, const String& from, const String& to);
		static void Split(const char* data, const char symbol, List<String>& result);
		static int32_t HasSubstring(const String& str1, const String& str2);
		static bool StartsWith(const char* str1, const char* str2);
		static bool StartsWith(const String& str1, const char* str2);
		static bool StartsWith(const String& str1, const String& str2);
		static bool EndsWith(const String& str1, const String& str2);
		static WString StringToWide(const String& str);
		static String WideToString(const WString& wstr);
		static String ToString(const std::filesystem::path& path);
		static String ToGenericString(const std::filesystem::path& path);
	};
}