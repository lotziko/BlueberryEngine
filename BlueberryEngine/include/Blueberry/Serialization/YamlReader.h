#pragma once

#include "Blueberry\Core\Base.h"

#include <iosfwd>

namespace Blueberry
{
	struct SerializationTree;

	class YamlReader
	{
	public:
		static void Read(List<SerializationTree>& trees, std::ifstream& stream, bool hasHeaders);
	};
}