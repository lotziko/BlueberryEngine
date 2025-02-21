#pragma once

namespace Blueberry
{
	class FileHelper
	{
	public:
		static void Save(const uint8_t* data, const size_t& length, const std::string& path);
		static void Load(uint8_t*& data, size_t& length, const std::string& path);
		static void Load(std::string& data, const std::string& path);
	};
}