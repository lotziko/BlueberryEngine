#include "bbpch.h"
#include "PathModifyCache.h"

#include "Editor\Path.h"

#include <fstream>

// TODO make it into an asset info cache containing also fileId of the main object and the list of fileIds+types in asset
// On creating of importer create dummy objects with corresponding fileIds and then pass them to deserializer when needed
namespace Blueberry
{
	std::unordered_map<std::string, long long> PathModifyCache::s_PathModifyCache = std::unordered_map<std::string, long long>();

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
				long long time;
				size_t pathSize;
				input.read((char*)&pathSize, sizeof(size_t));
				std::string path(pathSize, ' ');
				input.read(path.data(), pathSize);
				input.read((char*)&time, sizeof(long long));
				s_PathModifyCache.insert_or_assign(path, time);
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
			std::string path = pair.first;
			long long time = pair.second;
			size_t pathSize = path.size();
			output.write((char*)&pathSize, sizeof(size_t));
			output.write(path.data(), pathSize);
			output.write((char*)&time, sizeof(long long));
		}
		output.close();
	}

	long long PathModifyCache::Get(const std::string& path)
	{
		auto it = s_PathModifyCache.find(path);
		if (it != s_PathModifyCache.end())
		{
			return it->second;
		}
		return 0;
	}

	void PathModifyCache::Set(const std::string& path, const long long& lastWriteTime)
	{
		s_PathModifyCache.insert_or_assign(path, lastWriteTime);
	}
}
