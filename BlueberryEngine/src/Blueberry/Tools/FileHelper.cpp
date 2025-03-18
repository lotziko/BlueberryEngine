#include "bbpch.h"
#include "FileHelper.h"

namespace Blueberry
{
	void FileHelper::Save(const uint8_t* data, const size_t& length, const std::string& path)
	{
		auto file = fopen(path.c_str(), "wb");
		fwrite(data, sizeof(uint8_t) * length, 1, file);
		fclose(file);
	}

	void FileHelper::Load(uint8_t*& data, size_t& length, const std::string& path)
	{
		auto file = fopen(path.c_str(), "rb");
		fseek(file, 0, SEEK_END);
		length = ftell(file);
		rewind(file);
		data = BB_MALLOC_ARRAY(uint8_t, length);
		fread(data, sizeof(uint8_t) * length, 1, file);
		fclose(file);
	}

	void FileHelper::Load(std::string& data, const std::string& path)
	{
		size_t length;
		auto file = fopen(path.c_str(), "rb");
		fseek(file, 0, SEEK_END);
		length = ftell(file);
		rewind(file);
		data = std::string(length, ' ');
		fread(data.data(), sizeof(byte) * length, 1, file);
		fclose(file);
	}
}
