#include "Blueberry\Tools\StringHelper.h"

#include <cctype>

namespace Blueberry
{
	void StringHelper::Split(const char* data, const char symbol, List<String>& result)
	{
		String str(data);
		// Based on https://sentry.io/answers/split-string-in-cpp/
		// and https://stackoverflow.com/questions/19605720/error-in-strtok-split-using-c

		String::size_type pos = 0, f;
		while (((f = str.find(symbol, pos)) != String::npos))
		{
			result.push_back(str.substr(pos, f - pos));
			pos = f + 1;
		}
		if (pos < str.size())
		{
			result.push_back(str.substr(pos));
		}
	}

	// Based on https://stackoverflow.com/questions/3152241/case-insensitive-stdstring-find
	int32_t StringHelper::HasSubstring(const String& str1, const String& str2)
	{
		auto it = std::search(
			str1.begin(), str1.end(),
			str2.begin(), str2.end(),
			[](unsigned char ch1, unsigned char ch2) { return std::toupper(ch1) == std::toupper(ch2); }
		);
		return (it != str1.end());
	}
}