#pragma once

#include <map>
#include "rapidyaml\ryml.h"
#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	using FileId = uint64_t;

	struct SerializationContext
	{
		ryml::Tree tree;
		std::map<uint64_t, FileId> objectsFileIds;
		FileId maxId = 0;
	};
}