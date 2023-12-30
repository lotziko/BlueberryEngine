#include "Path.h"

namespace Blueberry
{
	std::filesystem::path Path::s_AssetsPath = "Assets";
	std::filesystem::path Path::s_DataPath = "Data";
	std::filesystem::path Path::s_AssetCachePath = "Data\\AssetCache";
	std::filesystem::path Path::s_ShaderCachePath = "Data\\ShaderCache";
	std::filesystem::path Path::s_TextureCachePath = "Data\\TextureCache";

	const std::filesystem::path& Path::GetAssetsPath()
	{
		return s_AssetsPath;
	}

	const std::filesystem::path& Path::GetDataPath()
	{
		return s_DataPath;
	}

	const std::filesystem::path& Path::GetAssetCachePath()
	{
		return s_AssetCachePath;
	}

	const std::filesystem::path& Path::GetShaderCachePath()
	{
		return s_ShaderCachePath;
	}

	const std::filesystem::path& Path::GetTextureCachePath()
	{
		return s_TextureCachePath;
	}

	void Path::SetProjectPath(const std::wstring& path)
	{
		auto assetsPath = path;
		assetsPath.append(L"\\Assets");
		s_AssetsPath = assetsPath;
		auto dataPath = path;
		dataPath.append(L"\\Data");
		s_DataPath = dataPath;
		auto assetCachePath = dataPath;
		assetCachePath.append(L"\\AssetCache");
		s_AssetCachePath = assetCachePath;
		auto shaderCachePath = dataPath;
		shaderCachePath.append(L"\\ShaderCache");
		s_ShaderCachePath = shaderCachePath;
		auto textureCachePath = dataPath;
		textureCachePath.append(L"\\TextureCache");
		s_TextureCachePath = textureCachePath;
	}
}