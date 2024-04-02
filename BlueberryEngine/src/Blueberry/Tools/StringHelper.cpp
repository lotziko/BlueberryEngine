#include "bbpch.h"
#include "StringHelper.h"

namespace Blueberry
{
	void StringHelper::Split(const char* data, const char symbol, std::vector<std::string>& result)
	{
		std::string str(data);
		// Based on https://sentry.io/answers/split-string-in-cpp/
		// and https://stackoverflow.com/questions/19605720/error-in-strtok-split-using-c

		std::string::size_type pos = 0, f;
		while (((f = str.find(symbol, pos)) != std::string::npos))
		{
			result.push_back(str.substr(pos, f - pos));
			pos = f + 1;
		}
		if (pos < str.size())
		{
			result.push_back(str.substr(pos));
		}
	}
}