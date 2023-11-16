#pragma once
#include <iomanip>
#include "yaml-cpp\yaml.h"
#include "Blueberry\Core\Guid.h"

namespace YAML
{
	template<>
	struct convert<Blueberry::Guid>
	{
		static Node encode(const Blueberry::Guid& rhs)
		{
			Node node;
			std::stringstream sstream;
			for (int i = 0; i < 2; i++)
			{
				sstream << std::setw(16) << std::setfill('0') << std::hex << rhs.data[i];
			}
			node.push_back(sstream.str());
			return node;
		}

		static bool decode(const Node& node, Blueberry::Guid& rhs)
		{
			if (!node.IsScalar())
				return false;
			
			std::stringstream sstream;
			std::string string = node.Scalar();
			for (int i = 0; i < 2; i++)
			{
				sstream.str(string.substr(i * 16, 16));
				sstream >> std::hex >> rhs.data[i];
			}
			return true;
		}
	};

	YAML::Emitter& operator<<(YAML::Emitter& out, Blueberry::Guid& guid)
	{
		std::stringstream sstream;
		for (int i = 0; i < 2; i++)
		{
			sstream << std::setw(16) << std::setfill('0') << std::hex << guid.data[i];
		}

		out << YAML::Flow;
		out << sstream.str();
		return out;
	}
}