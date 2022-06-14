#include "bbpch.h"
#include "Object.h"

namespace Blueberry
{
	const std::size_t Object::Type = std::hash<std::string>()(TO_STRING(Object));
}