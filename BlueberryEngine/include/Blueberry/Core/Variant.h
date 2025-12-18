#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\ObjectPtr.h"

#include <variant>

namespace Blueberry
{
	// Changing arguments requires changing serializers
	using Variant = std::variant<bool, int32_t, uint32_t, int64_t, uint64_t, float, String, Vector2, Vector2Int, Vector3, Vector3Int, Vector4, Vector4Int, Quaternion, Color, ObjectPtr<Object>>;
}