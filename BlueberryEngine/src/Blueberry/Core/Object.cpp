#include "bbpch.h"
#include "Object.h"

namespace Blueberry
{
	const std::size_t Object::Type = std::hash<std::string>()(TO_STRING(Object));
	const std::size_t Object::ParentType = 0;

	bool Object::IsClassType(const std::size_t classType) const
	{
		return classType == Type;
	}

	std::size_t Object::GetType() const
	{
		return Type;
	}

	std::string Object::ToString() const
	{
		return "Object";
	}
}