#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class InspectorExpandedItemsCache
	{
	public:
		static void Load();
		static void Save();

		static bool Get(const String& name);
		static void Set(const String& name, const bool& expanded);

	private:
		static HashSet<String> s_InspectorExpandedItemsCache;
	};
}