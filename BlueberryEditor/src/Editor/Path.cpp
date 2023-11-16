#include "Path.h"

namespace Blueberry
{
	std::filesystem::path Path::s_AssetsPath = "Assets";
	std::filesystem::path Path::s_DataPath = "Data";

	const std::filesystem::path& Path::GetAssetsPath()
	{
		return s_AssetsPath;
	}

	const std::filesystem::path& Path::GetDataPath()
	{
		return s_DataPath;
	}

	void Path::SetProjectPath(const std::wstring& path)
	{
		auto assetsPath = path;
		assetsPath.append(L"\\Assets");
		s_AssetsPath = assetsPath;
		auto dataPath = path;
		dataPath.append(L"\\Data");
		s_DataPath = dataPath;
	}
}