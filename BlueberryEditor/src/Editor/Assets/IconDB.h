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
		static void Register(const TypeId& type, const String& path);

	private:
		static Dictionary<TypeId, Texture*> s_AssetIcons;
	};

#define REGISTER_ICON( objectType, path ) IconDB::Register(objectType, path);
}