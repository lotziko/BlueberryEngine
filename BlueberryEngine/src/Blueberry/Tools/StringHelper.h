#pragma once

namespace Blueberry
{
	class StringHelper
	{
	public:
		static void Split(const char* data, const char symbol, List<std::string>& result);
		static int32_t HasSubstring(const std::string& str1, const std::string& str2);
	};
}