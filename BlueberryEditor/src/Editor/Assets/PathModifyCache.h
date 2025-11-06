#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	struct WriteInfo
	{
		long long assetLastWrite;
		long long metaLastWrite;
	};

	class PathModifyCache
	{
	public:
		static void Load();
		static void Save();

		static WriteInfo Get(const String& path);
		static void Set(const String& path, const WriteInfo& writeInfo);

	private:
		static Dictionary<String, WriteInfo> s_PathModifyCache;
	};
}