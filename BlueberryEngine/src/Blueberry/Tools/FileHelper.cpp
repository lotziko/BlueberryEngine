#include "Blueberry\Tools\FileHelper.h"

namespace Blueberry
{
	void FileHelper::Save(const uint8_t* data, const size_t& length, const String& path)
	{
		auto file = fopen(path.c_str(), "wb");
		fwrite(data, sizeof(uint8_t) * length, 1, file);
		fclose(file);
	}

	void FileHelper::Load(uint8_t*& data, size_t& length, const String& path)
	{
		auto file = fopen(path.c_str(), "rb");
		fseek(file, 0, SEEK_END);
		length = ftell(file);
		rewind(file);
		data = BB_MALLOC_ARRAY(uint8_t, length);
		fread(data, sizeof(uint8_t) * length, 1, file);
		fclose(file);
	}

	void FileHelper::Load(String& data, const String& path)
	{
		size_t length;
		auto file = fopen(path.c_str(), "rb");
		fseek(file, 0, SEEK_END);
		length = ftell(file);
		rewind(file);
		data = String(length, ' ');
		fread(data.data(), sizeof(uint8_t) * length, 1, file);
		fclose(file);
	}
}
