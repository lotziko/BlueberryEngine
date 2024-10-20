#pragma once

namespace Blueberry
{
	class Texture;
	class Object;

	class IconDB
	{
	public:
		static Texture* GetAssetIcon(Object* asset);
		static void Register(const std::size_t& type, const std::string& path);

	private:
		static std::unordered_map<std::size_t, Texture*> s_AssetIcons;
	};

#define REGISTER_ICON( objectType, path ) IconDB::Register(objectType, path);
}