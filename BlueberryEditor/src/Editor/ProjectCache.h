#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	struct ProjectInfo
	{
		WString wpath;
		String path;
	};

	class ProjectCache
	{
	public:
		static void Load();
		static void Save();

		static const List<ProjectInfo>& Get();
		static void Add(const String& path);
		static void Remove(const String& path);

	private:
		static List<ProjectInfo> s_ProjectInfoCache;
	};
}