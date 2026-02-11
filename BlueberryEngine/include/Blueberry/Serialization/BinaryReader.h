#pragma once

#include "Blueberry\Core\Base.h"

#include <iosfwd>

namespace Blueberry
{
	struct SerializationTree;

	class BinaryReader
	{
	public:
		static void Read(List<SerializationTree>& trees, std::ifstream& stream);
	};
}