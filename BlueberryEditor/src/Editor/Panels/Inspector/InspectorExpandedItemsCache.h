#pragma once

namespace Blueberry
{
	class InspectorExpandedItemsCache
	{
	public:
		static void Load();
		static void Save();

		static bool Get(const std::string& name);
		static void Set(const std::string& name, const bool& expanded);

	private:
		static HashSet<std::string> s_InspectorExpandedItemsCache;
	};
}