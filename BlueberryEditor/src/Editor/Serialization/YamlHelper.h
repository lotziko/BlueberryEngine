#pragma once

#include "Blueberry\Core\Base.h"

#include <rapidyaml\ryml.h>

namespace Blueberry
{
	class YamlHelper
	{
	public:
		static void Save(ryml::Tree& tree, const String& path);
		static void Load(ryml::Tree& tree, const String& path);
	};
}