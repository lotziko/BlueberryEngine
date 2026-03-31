#include "DependencyCache.h"

#include "Editor\Path.h"

#include <fstream>

namespace Blueberry
{
	Dictionary<Guid, HashSet<Guid>> DependencyCache::s_DependencyCache = {};
	Dictionary<Guid, HashSet<Guid>> DependencyCache::s_ReverseDependencyCache = {};

	void DependencyCache::Load()
	{
		std::filesystem::path dataPath = Path::GetAssetCachePath();
		dataPath.append("DependencyCache");

		if (std::filesystem::exists(dataPath))
		{
			std::ifstream input;
			input.open(dataPath, std::ifstream::binary);
			size_t cacheSize;
			input.read(reinterpret_cast<char*>(&cacheSize), sizeof(size_t));
			for (size_t i = 0; i < cacheSize; ++i)
			{
				Guid guid;
				input.read(reinterpret_cast<char*>(&guid), sizeof(Guid));
				size_t dependencyCount;
				input.read(reinterpret_cast<char*>(&dependencyCount), sizeof(size_t));
				for (size_t j = 0; j < dependencyCount; ++j)
				{
					Guid dependency;
					input.read(reinterpret_cast<char*>(&dependency), sizeof(Guid));
					s_DependencyCache[guid].insert(dependency);
					s_ReverseDependencyCache[dependency].insert(guid);
				}
			}
		}
	}

	void DependencyCache::Save()
	{
		std::filesystem::path dataPath = Path::GetAssetCachePath();
		dataPath.append("DependencyCache");

		size_t cacheSize = s_DependencyCache.size();
		std::ofstream output;
		output.open(dataPath, std::ofstream::binary);
		output.write(reinterpret_cast<char*>(&cacheSize), sizeof(size_t));

		for (auto& pair : s_DependencyCache)
		{
			Guid guid = pair.first;
			output.write(reinterpret_cast<char*>(&guid), sizeof(Guid));
			size_t dependencyCount = pair.second.size();
			output.write(reinterpret_cast<char*>(&dependencyCount), sizeof(size_t));
			for (Guid dependency : pair.second)
			{
				output.write(reinterpret_cast<char*>(&dependency), sizeof(Guid));
			}
		}
	}

	void DependencyCache::Get(const Guid& assetGuid, HashSet<Guid>& dependent)
	{
		auto it = s_ReverseDependencyCache.find(assetGuid);
		if (it != s_ReverseDependencyCache.end())
		{
			for (Guid dependency : it->second)
			{
				dependent.insert(dependency);
			}
		}
	}

	void DependencyCache::Set(const Guid& guid, const HashSet<Guid>& dependencies)
	{
		for (Guid dependency : dependencies)
		{
			s_DependencyCache[guid].insert(dependency);
			s_ReverseDependencyCache[dependency].insert(guid);
		}
	}
}
