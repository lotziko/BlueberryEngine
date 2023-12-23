#pragma once

namespace Blueberry
{
	class FileHelper
	{
	public:
		static void Save(byte* data, const size_t& length, const std::string& path);
	};
}