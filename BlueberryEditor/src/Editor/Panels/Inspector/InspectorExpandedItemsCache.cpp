#include "bbpch.h"
#include "InspectorExpandedItemsCache.h"

#include "Editor\Path.h"

#include <fstream>

namespace Blueberry
{
	std::unordered_set<std::string> InspectorExpandedItemsCache::s_InspectorExpandedItemsCache = std::unordered_set<std::string>();

	void InspectorExpandedItemsCache::Load()
	{
		auto dataPath = Path::GetDataPath();
		dataPath.append("InspectorExpandedItemsCache");

		if (std::filesystem::exists(dataPath))
		{
			std::ifstream input;
			input.open(dataPath, std::ifstream::binary);

			std::string line;
			while (std::getline(input, line))
			{
				s_InspectorExpandedItemsCache.insert(line);
			}
			input.close();
		}
	}

	void InspectorExpandedItemsCache::Save()
	{
		auto dataPath = Path::GetDataPath();
		dataPath.append("InspectorExpandedItemsCache");

		std::ofstream output;
		output.open(dataPath, std::ofstream::binary);

		for (auto& item : s_InspectorExpandedItemsCache)
		{
			output << item << std::endl;
		}
		output.close();
	}

	bool InspectorExpandedItemsCache::Get(const std::string& name)
	{
		return s_InspectorExpandedItemsCache.count(name) > 0;
	}

	void InspectorExpandedItemsCache::Set(const std::string& name, const bool& expanded)
	{
		if (expanded)
		{
			s_InspectorExpandedItemsCache.insert(name);
		}
		else
		{
			s_InspectorExpandedItemsCache.erase(name);
		}
	}
}
