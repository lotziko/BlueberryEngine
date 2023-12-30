#pragma once

#include <filesystem>

namespace Blueberry
{
	class Path
	{
	public:
		static const std::filesystem::path& GetAssetsPath();
		static const std::filesystem::path& GetDataPath();
		static const std::filesystem::path& GetAssetCachePath();
		static const std::filesystem::path& GetShaderCachePath();
		static const std::filesystem::path& GetTextureCachePath();
		static void SetProjectPath(const std::wstring& path);

	private:
		static std::filesystem::path s_AssetsPath;
		static std::filesystem::path s_DataPath;
		static std::filesystem::path s_AssetCachePath;
		static std::filesystem::path s_ShaderCachePath;
		static std::filesystem::path s_TextureCachePath;
	};
}