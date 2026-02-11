#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Guid.h"

namespace Blueberry
{
	class DependencyCache
	{
	public:
		static void Load();
		static void Save();

		static void Get(const Guid& assetGuid, HashSet<Guid>& dependent);
		static void Set(const Guid& guid, const HashSet<Guid>& dependencies);

	private:
		static Dictionary<Guid, HashSet<Guid>> s_DependencyCache;
		static Dictionary<Guid, HashSet<Guid>> s_ReverseDependencyCache;
	};
}