#include "Path.h"

#include "Blueberry\Tools\StringHelper.h"

namespace Blueberry
{
	std::filesystem::path Path::s_AssetsPath = "Assets";
	std::filesystem::path Path::s_DataPath = "Data";
	std::filesystem::path Path::s_AssemblyPath = "Source";
	std::filesystem::path Path::s_AssetCachePath = "Data\\AssetCache";
	std::filesystem::path Path::s_ShaderCachePath = "Data\\ShaderCache";
	std::filesystem::path Path::s_TextureCachePath = "Data\\TextureCache";
	std::filesystem::path Path::s_PhysicsShapeCachePath = "Data\\PhysicsShapeCache";
	std::filesystem::path Path::s_ThumbnailCachePath = "Data\\ThumbnailCache";
	std::filesystem::path Path::s_BuildPath = "Build";

	const std::filesystem::path& Path::GetAssetsPath()
	{
		return s_AssetsPath;
	}

	const std::filesystem::path& Path::GetDataPath()
	{
		return s_DataPath;
	}

	const std::filesystem::path& Path::GetAssemblyPath()
	{
		return s_AssemblyPath;
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

	const std::filesystem::path& Path::GetPhysicsShapeCachePath()
	{
		return s_PhysicsShapeCachePath;
	}

	const std::filesystem::path& Path::GetThumbnailCachePath()
	{
		return s_ThumbnailCachePath;
	}

	const std::filesystem::path& Path::GetBuildPath()
	{
		return s_BuildPath;
	}

	void Path::SetProjectPath(const WString& path)
	{
		std::filesystem::path assetsPath = path;
		assetsPath += "\\Assets";
		s_AssetsPath = assetsPath;
		std::filesystem::path dataPath = path;
		dataPath += "\\Data";
		s_DataPath = dataPath;
		std::filesystem::path assemblyPath = path;
		assemblyPath += "\\Source";
		s_AssemblyPath = assemblyPath;
		std::filesystem::path assetCachePath = dataPath;
		assetCachePath += "\\AssetCache";
		s_AssetCachePath = assetCachePath;
		std::filesystem::path shaderCachePath = dataPath;
		shaderCachePath += "\\ShaderCache";
		s_ShaderCachePath = shaderCachePath;
		std::filesystem::path textureCachePath = dataPath;
		textureCachePath += "\\TextureCache";
		s_TextureCachePath = textureCachePath;
		std::filesystem::path physicsShapeCachePath = dataPath;
		physicsShapeCachePath += "\\PhysicsShapeCache";
		s_PhysicsShapeCachePath = physicsShapeCachePath;
		std::filesystem::path thumbnailCachePath = dataPath;
		thumbnailCachePath += "\\ThumbnailCache";
		s_ThumbnailCachePath = thumbnailCachePath;
		std::filesystem::path buildPath = path;
		buildPath += "\\Build";
		s_BuildPath = buildPath;
	}

	std::filesystem::path Path::GetAssetPath(const String& relativePath)
	{
		std::filesystem::path assetPath = s_AssetsPath;
		assetPath.append(relativePath);
		return assetPath;
	}

	String Path::GetAssetsPath(const String& relativePath)
	{
		std::filesystem::path dataPath = s_AssetsPath;
		dataPath.append(relativePath);
		return StringHelper::ToString(dataPath);
	}

	String Path::GetAssetCachePath(const String& relativePath)
	{
		std::filesystem::path dataPath = s_AssetCachePath;
		dataPath.append(relativePath);
		return StringHelper::ToString(dataPath);
	}
}