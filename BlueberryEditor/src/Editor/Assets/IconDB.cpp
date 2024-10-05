#include "bbpch.h"
#include "IconDB.h"

#include "Blueberry\Assets\AssetLoader.h"

namespace Blueberry
{
	std::unordered_map<size_t, Texture*> IconDB::s_AssetIcons = {};

	Texture* IconDB::GetAssetIcon(Object* asset)
	{
		auto iconIt = s_AssetIcons.find(asset->GetType());
		if (iconIt != s_AssetIcons.end())
		{
			return iconIt->second;
		}
		return s_AssetIcons[Object::Type];
	}

	void IconDB::Register(const size_t& type, const std::string& path)
	{
		s_AssetIcons.insert_or_assign(type, (Texture*)AssetLoader::Load(path));
	}
}