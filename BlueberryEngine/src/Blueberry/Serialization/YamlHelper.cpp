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
		// https://stackoverflow.com/questions/2602013/read-whole-ascii-file-into-c-stdstring
		std::ifstream t;
		int length;
		t.open(path.c_str());
		t.seekg(0, std::ios::end);
		length = t.tellg();
		t.seekg(0, std::ios::beg);
		char* buffer = new char[length];
		t.read(buffer, length);
		t.close();
		tree = ryml::parse_in_arena(ryml::to_csubstr(buffer));
		delete[] buffer;
	}
}
