#include "PathModifyCache.h"

#include "Editor\Path.h"

#include <fstream>

// TODO make it into an asset info cache containing also fileId of the main object and the list of fileIds+types in asset
// On creating of importer create dummy objects with corresponding fileIds and then pass them to deserializer when needed
namespace Blueberry
{
	Dictionary<String, WriteInfo> PathModifyCache::s_PathModifyCache = {};

	void PathModifyCache::Load()
	{
		auto dataPath = Path::GetAssetCachePath();
		dataPath.append("PathModifyCache");

		if (std::filesystem::exists(dataPath))
		{
			std::ifstream input;
			input.open(dataPath, std::ifstream::binary);
			size_t cacheSize;
			input.read((char*)&cacheSize, sizeof(size_t));

			for (size_t i = 0; i < cacheSize; ++i)
			{
				WriteInfo writeInfo;
				size_t pathSize;
				input.read((char*)&pathSize, sizeof(size_t));
				String path(pathSize, ' ');
				input.read(path.data(), pathSize);
				input.read((char*)&writeInfo.assetLastWrite, sizeof(long long));
				input.read((char*)&writeInfo.metaLastWrite, sizeof(long long));
				s_PathModifyCache.insert_or_assign(path, writeInfo);
			}
			input.close();
		}
	}

	void PathModifyCache::Save()
	{
		auto dataPath = Path::GetAssetCachePath();
		dataPath.append("PathModifyCache");

		size_t cacheSize = s_PathModifyCache.size();
		std::ofstream output;
		output.open(dataPath, std::ofstream::binary);
		output.write((char*)&cacheSize, sizeof(size_t));

		for (auto& pair : s_PathModifyCache)
		{
			String path = pair.first;
			long long assetLastWrite = pair.second.assetLastWrite;
			long long metaLastWrite = pair.second.metaLastWrite;
			size_t pathSize = path.size();
			output.write((char*)&pathSize, sizeof(size_t));
			output.write(path.data(), pathSize);
			output.write((char*)&assetLastWrite, sizeof(long long));
			output.write((char*)&metaLastWrite, sizeof(long long));
		}
		output.close();
	}

	WriteInfo PathModifyCache::Get(const String& path)
	{
		auto it = s_PathModifyCache.find(path);
		if (it != s_PathModifyCache.end())
		{
			return it->second;
		}
		return WriteInfo();
	}

	void PathModifyCache::Set(const String& path, const WriteInfo& writeInfo)
	{
		s_PathModifyCache.insert_or_assign(path, writeInfo);
	}
}
