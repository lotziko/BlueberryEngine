#include "PathHelper.h"

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
}
