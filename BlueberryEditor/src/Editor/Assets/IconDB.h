#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Texture;
	class Object;

	class IconDB
	{
	public:
		static Texture* GetAssetIcon(Object* asset);
		static void Register(const size_t& type, const String& path);

	private:
		static Dictionary<size_t, Texture*> s_AssetIcons;
	};

#define REGISTER_ICON( objectType, path ) IconDB::Register(objectType, path);
}