#include "bbpch.h"
#include "FileHelper.h"

namespace Blueberry
{
	void FileHelper::Save(const byte* data, const size_t& length, const std::string& path)
	{
		auto file = fopen(path.c_str(), "wb");
		fwrite(data, sizeof(byte) * length, 1, file);
		fclose(file);
	}

	void FileHelper::Load(byte*& data, size_t& length, const std::string& path)
	{
		auto file = fopen(path.c_str(), "rb");
		fseek(file, 0, SEEK_END);
		length = ftell(file);
		rewind(file);
		data = (byte*)malloc(sizeof(byte)*length);
		fread(data, sizeof(byte) * length, 1, file);
		fclose(file);
	}
}
