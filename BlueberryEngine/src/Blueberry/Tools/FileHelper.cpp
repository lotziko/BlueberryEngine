#include "Blueberry\Tools\FileHelper.h"

namespace Blueberry
{
	void FileHelper::Save(const uint8_t* data, size_t length, const String& path)
	{
		auto file = fopen(path.c_str(), "wb");
		fwrite(data, sizeof(uint8_t) * length, 1, file);
		fclose(file);
	}

	void FileHelper::Save(const List<uint8_t>& data, const String& path)
	{
		auto file = fopen(path.c_str(), "wb");
		fwrite(data.data(), sizeof(uint8_t) * data.size(), 1, file);
		fclose(file);
	}

	List<uint8_t> FileHelper::LoadBinary(const String& path)
	{
		List<uint8_t> data;
		auto file = fopen(path.c_str(), "rb");
		fseek(file, 0, SEEK_END);
		long length = ftell(file);
		rewind(file);
		data.resize(length);
		fread(data.data(), sizeof(uint8_t) * length, 1, file);
		fclose(file);
		return data;
	}

	String FileHelper::LoadText(const String& path)
	{
		size_t length;
		auto file = fopen(path.c_str(), "rb");
		fseek(file, 0, SEEK_END);
		length = ftell(file);
		rewind(file);
		String data = String(length, ' ');
		fread(data.data(), sizeof(uint8_t) * length, 1, file);
		fclose(file);
		return data;
	}
}
