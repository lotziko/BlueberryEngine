#include "bbpch.h"
#include "YamlHelper.h"

#include "rapidyaml\ryml.h"
#include <fstream>

namespace Blueberry
{
	void YamlHelper::Save(ryml::Tree& tree, const std::string& path)
	{
		auto file = fopen(path.c_str(), "w");
		ryml::emit_yaml(tree, tree.root_id(), file);
		fclose(file);
	}
	
	void YamlHelper::Load(ryml::Tree& tree, const std::string& path)
	{
		auto file = fopen(path.c_str(), "r");
		fseek(file, 0, SEEK_END);
		size_t length = ftell(file);
		rewind(file);
		std::string data(length, ' ');
		fread(data.data(), sizeof(char) * length, 1, file);
		fclose(file);
		tree = ryml::parse_in_arena(ryml::csubstr(data.data(), length));
		tree.resolve_tags();
	}
}
