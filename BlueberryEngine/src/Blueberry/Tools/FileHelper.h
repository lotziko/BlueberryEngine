#pragma once

namespace Blueberry
{
	class FileHelper
	{
	public:
		static void Save(const byte* data, const size_t& length, const std::string& path);
		static void Load(byte*& data, size_t& length, const std::string& path);
	};
}