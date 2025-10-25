#pragma once

#include "Blueberry\Core\Base.h"

#include <filesystem>

namespace Blueberry
{
	class PathHelper
	{
	public:
		static long long GetLastWriteTime(const std::filesystem::path& path);
		static long long GetLastWriteTime(const String& path);
		static long long GetDirectoryLastWriteTime(const String& path);
	};
}