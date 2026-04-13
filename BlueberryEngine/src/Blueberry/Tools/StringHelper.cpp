#include "Blueberry\Tools\StringHelper.h"

#include <cctype>
#include <codecvt>
#include <filesystem>

namespace Blueberry
{
	void StringHelper::Replace(String& str, const String& from, const String& to)
	{
		// Based on https://medium.com/@nerudaj/tuesday-coding-tip-53-replace-all-occurrences-of-substring-in-std-string-99a4181cbb24
		size_t pos = str.find(from, 0);
		while (pos != std::string::npos)
		{
			str.replace(pos, from.length(), to);
			pos = str.find(from, pos + to.length());
		}
	}

	void StringHelper::Split(const char* data, const char symbol, List<String>& result)
	{
		// Based on https://sentry.io/answers/split-string-in-cpp/
		// and https://stackoverflow.com/questions/19605720/error-in-strtok-split-using-c
		String str(data);
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

	int32_t StringHelper::HasSubstring(const String& str1, const String& str2)
	{
		// Based on https://stackoverflow.com/questions/3152241/case-insensitive-stdstring-find
		auto it = std::search(
			str1.begin(), str1.end(),
			str2.begin(), str2.end(),
			[](unsigned char ch1, unsigned char ch2) { return std::toupper(ch1) == std::toupper(ch2); }
		);
		return (it != str1.end());
	}

	bool StringHelper::EndsWith(const String& str1, const String& str2)
	{
		// Based on https://www.geeksforgeeks.org/cpp/check-if-string-ends-substring-in-cpp/
		if (str2.size() > str1.size())
		{
			return false;
		}
		return str1.compare(str1.size() - str2.size(), str2.size(), str2) == 0;
	}

	WString StringHelper::StringToWide(const String& str)
	{
		// Based on https://stackoverflow.com/questions/215963/how-do-you-properly-use-widechartomultibyte
		if (str.empty())
		{
			return L"";
		}
		int sizeNeeded = MultiByteToWideChar(CP_UTF8, 0, &str[0], static_cast<int>(str.size()), NULL, 0);
		WString wstrTo(sizeNeeded, 0);
		MultiByteToWideChar(CP_UTF8, 0, &str[0], static_cast<int>(str.size()), &wstrTo[0], sizeNeeded);
		return wstrTo;
	}

	String StringHelper::WideToString(const WString& wstr)
	{
		// Based on https://stackoverflow.com/questions/215963/how-do-you-properly-use-widechartomultibyte
		if (wstr.empty())
		{
			return "";
		}
		int sizeNeeded = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], static_cast<int>(wstr.size()), NULL, 0, NULL, NULL);
		String strTo(sizeNeeded, 0);
		WideCharToMultiByte(CP_UTF8, 0, &wstr[0], static_cast<int>(wstr.size()), &strTo[0], sizeNeeded, NULL, NULL);
		return strTo;
	}

	String StringHelper::ToString(const std::filesystem::path& path)
	{
		return WideToString(path.string<wchar_t, std::char_traits<wchar_t>, STLAllocator<wchar_t>>(STLAllocator<wchar_t>()));
	}

	String StringHelper::ToGenericString(const std::filesystem::path& path)
	{
		return path.generic_string<char, std::char_traits<char>, STLAllocator<char>>(STLAllocator<char>());
	}
}