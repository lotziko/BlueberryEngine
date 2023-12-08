#pragma once

#include <rapidyaml\ryml.h>

namespace Blueberry
{
	class YamlHelper
	{
	public:
		static void Save(ryml::Tree& tree, const std::string& path);
		static void Load(ryml::Tree& tree, const std::string& path);
	};
}