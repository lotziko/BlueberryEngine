#pragma once

namespace Blueberry
{
	class PathModifyCache
	{
	public:
		static void Load();
		static void Save();

		static long long Get(const std::string& path);
		static void Set(const std::string& path, const long long& lastWriteTime);

	private:
		static std::map<std::string, long long> s_PathModifyCache;
	};
}