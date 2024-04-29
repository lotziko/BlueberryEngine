#include "bbpch.h"
#include "ModifyCache.h"

#include "Editor\Path.h"

#include <fstream>
#include <sstream>

// TODO make it into an asset info cache containing also fileId of the main object and the list of fileIds in asset
namespace Blueberry
{
	std::map<std::string, long long> ModifyCache::s_PathModifyCache = std::map<std::string, long long>();

	void ModifyCache::Load()
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
				size_t pathSize;
				input.read((char*)&pathSize, sizeof(size_t));
				std::string path(pathSize, ' ');
				input.read(path.data(), pathSize);
				long long time;
				input.read((char*)&time, sizeof(long long));
				s_PathModifyCache.insert_or_assign(path, time);
			}
			input.close();
		}
	}

	void ModifyCache::Save()
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

	long long ModifyCache::Get(const std::string& path)
	{
		auto it = s_PathModifyCache.find(path);
		if (it != s_PathModifyCache.end())
		{
			return it->second;
		}
		return 0;
	}

	void ModifyCache::Set(const std::string& path, const long long& lastWriteTime)
	{
		s_PathModifyCache.insert_or_assign(path, lastWriteTime);
	}
}
