#pragma once

namespace Blueberry
{
	class FileHelper
	{
	public:
		static void Save(BYTE* data, const size_t& length, const std::string& path);
	};
}