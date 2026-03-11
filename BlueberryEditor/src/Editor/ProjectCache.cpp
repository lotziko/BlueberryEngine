#include "ProjectCache.h"

#include "Editor\Misc\PlatformHelper.h"

#include <fstream>
#include <filesystem>
#include <codecvt>

namespace Blueberry
{
	List<ProjectInfo> ProjectCache::s_ProjectInfoCache = {};

	String ConvertString(const WString& wstr)
	{
		static std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
		return String(converter.to_bytes(wstr.c_str()));
	}

	void ProjectCache::Load()
	{
		String path = PlatformHelper::GetEditorDataFolder().append("Projects");
		if (std::filesystem::exists(path))
		{
			std::wifstream input;
			input.open(path.c_str(), std::wifstream::binary);
			std::wstring line;
			while (std::getline(input, line))
			{
				ProjectInfo info;
				info.wpath = WString(line);
				info.path = ConvertString(info.wpath);
				s_ProjectInfoCache.push_back(std::move(info));
			}
			input.close();
		}
	}

	void ProjectCache::Save()
	{
		String folder = PlatformHelper::GetEditorDataFolder();
		if (!std::filesystem::exists(folder))
		{
			std::filesystem::create_directories(folder);
		}
		String path = folder.append("Projects");

		size_t cacheSize = s_ProjectInfoCache.size();
		std::wofstream output;
		output.open(path.c_str(), std::wofstream::binary);
		for (auto& info : s_ProjectInfoCache)
		{
			output << info.wpath;
		}
		output.close();
	}

	const List<ProjectInfo>& ProjectCache::Get()
	{
		return s_ProjectInfoCache;
	}

	void ProjectCache::Add(const WString& path)
	{
		for (auto& info : s_ProjectInfoCache)
		{
			if (info.wpath == path)
			{
				return;
			}
		}
		ProjectInfo info;
		info.wpath = path;
		info.path = ConvertString(path);
		s_ProjectInfoCache.push_back(std::move(info));
	}

	void ProjectCache::Remove(const WString& path)
	{
		for (auto& it = s_ProjectInfoCache.begin(); it != s_ProjectInfoCache.end(); ++it)
		{
			if (it->wpath == path)
			{
				s_ProjectInfoCache.erase(it);
				break;
			}
		}
	}
}
