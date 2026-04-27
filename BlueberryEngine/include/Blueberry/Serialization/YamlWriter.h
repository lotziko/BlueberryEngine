#pragma once

#include "Blueberry\Core\Base.h"

#include <iosfwd>

namespace Blueberry
{
	struct SerializationTree;

	class YamlWriter
	{
	public:
		static void Write(List<SerializationTree>& trees, std::ofstream& stream, bool hasHeaders);
	};
}