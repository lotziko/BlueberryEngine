#pragma once

namespace Blueberry
{
	class StringHelper
	{
	public:
		static void Split(const char* data, const char symbol, std::vector<std::string>& result);
	};
}