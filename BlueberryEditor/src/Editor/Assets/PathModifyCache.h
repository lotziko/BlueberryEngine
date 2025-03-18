#pragma once

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

		static WriteInfo Get(const std::string& path);
		static void Set(const std::string& path, const WriteInfo& writeInfo);

	private:
		static Dictionary<std::string, WriteInfo> s_PathModifyCache;
	};
}