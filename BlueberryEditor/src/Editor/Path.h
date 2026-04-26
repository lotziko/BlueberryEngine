#pragma once

#include "Blueberry\Core\Base.h"

#include <filesystem>

namespace Blueberry
{
	class Path
	{
	public:
		static const std::filesystem::path& GetAssetsPath();
		static const std::filesystem::path& GetDataPath();
		static const std::filesystem::path& GetAssemblyPath();
		static const std::filesystem::path& GetAssetCachePath();
		static const std::filesystem::path& GetShaderCachePath();
		static const std::filesystem::path& GetTextureCachePath();
		static const std::filesystem::path& GetPhysicsShapeCachePath();
		static const std::filesystem::path& GetAudioCachePath();
		static const std::filesystem::path& GetThumbnailCachePath();
		static const std::filesystem::path& GetBuildPath();
		static const std::filesystem::path& GetProjectPath();
		static void SetProjectPath(const WString& path);

		static std::filesystem::path GetAssetPath(const String& relativePath);
		static String GetAssetsPath(const String& relativePath);
		static String GetAssetCachePath(const String& relativePath);

	private:
		static std::filesystem::path s_ProjectPath;
		static std::filesystem::path s_AssetsPath;
		static std::filesystem::path s_DataPath;
		static std::filesystem::path s_AssemblyPath;
		static std::filesystem::path s_AssetCachePath;
		static std::filesystem::path s_ShaderCachePath;
		static std::filesystem::path s_TextureCachePath;
		static std::filesystem::path s_PhysicsShapeCachePath;
		static std::filesystem::path s_AudioCachePath;
		static std::filesystem::path s_ThumbnailCachePath;
		static std::filesystem::path s_BuildPath;
	};
}