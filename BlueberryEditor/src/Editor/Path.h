#pragma once

#include <filesystem>

namespace Blueberry
{
	class Path
	{
	public:
		static const std::filesystem::path& GetAssetsPath();
		static const std::filesystem::path& GetDataPath();
		static void SetProjectPath(const std::wstring& path);

	private:
		static std::filesystem::path s_AssetsPath;
		static std::filesystem::path s_DataPath;
	};
}