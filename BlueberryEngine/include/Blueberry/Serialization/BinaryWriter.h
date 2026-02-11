#pragma once

#include "Blueberry\Core\Base.h"

#include <iosfwd>

namespace Blueberry
{
	struct KeyData;
	struct SerializationTree;

	class BinaryWriter
	{
	public:
		static void Write(List<SerializationTree>& trees, std::ofstream& stream);
	};
}