#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	enum class BindingType;
	struct ClassInfo;
	struct FieldInfo;

	// Serializer should iterate all fields and put data into nodes
	// Need to have a callback that will check the type and put additional nodes for prefab references

	struct SerializationNode
	{
		FieldInfo* fieldInfo; // will not work with hot reload
		List<uint8_t> value;
		List<std::unique_ptr<SerializationNode>> children;
	};

	struct SerializationObjectNode
	{
		ClassInfo* classInfo; // will not work with hot reload
		ObjectId id;
		List<std::unique_ptr<SerializationNode>> fields;
	};

	struct SerializationTree
	{
		List<std::unique_ptr<SerializationObjectNode>> objects;
	};
}