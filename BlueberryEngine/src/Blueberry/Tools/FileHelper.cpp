#include "bbpch.h"
#include "FileHelper.h"

namespace Blueberry
{
	void FileHelper::Save(byte* data, const size_t& length, const std::string& path)
	{
		auto file = fopen(path.c_str(), "w");
		fwrite(data, sizeof(byte) * length, 1, file);
		fclose(file);
	}
}
