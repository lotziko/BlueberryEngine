#include "PathHelper.h"

#include "Blueberry\Tools\ByteConverter.h"
#include "Blueberry\Tools\StringHelper.h"

#include <cstdlib>
#include <fstream>

namespace Blueberry
{
	long long PathHelper::GetLastWriteTime(const std::filesystem::path& path)
	{
		return std::chrono::duration_cast<std::chrono::seconds>(std::filesystem::last_write_time(path).time_since_epoch()).count();
	}

	long long PathHelper::GetLastWriteTime(const String& path)
	{
		return std::chrono::duration_cast<std::chrono::seconds>(std::filesystem::last_write_time(path).time_since_epoch()).count();
	}

	long long PathHelper::GetDirectoryLastWriteTime(const String& path)
	{
		long long result = 0;
		for (const auto& entry : std::filesystem::recursive_directory_iterator(path))
		{
			result = std::max(result, std::chrono::duration_cast<std::chrono::seconds>(std::filesystem::last_write_time(entry.path()).time_since_epoch()).count());
		}
		return result;
	}

	Guid PathHelper::GetMetaGuid(const String& path)
	{
		Guid guid = {};
		std::ifstream input;
		input.open(path.data(), std::ios::in | std::ofstream::binary);
		if (input.is_open())
		{
			String line;
			std::getline(input, line);
			if (StringHelper::StartsWith(line, "Guid:"))
			{
				size_t colonPos = line.find(':');
				size_t guidPos = line.find_first_not_of(' ', colonPos + 1);
				size_t guidEndPos = line.find_first_of(" \t\f\v\n\r", guidPos + 1);
				String guidString = line.substr(guidPos, guidEndPos == std::string::npos ? std::string::npos : (guidEndPos - guidPos));
				ByteConverter::HexStringToBytes(guidString.data(), guid.data, guidString.size());
			}
			input.close();
		}
		return guid;
	}
}
