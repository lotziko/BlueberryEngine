#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class MathHelper
	{
	public:
		static bool Intersects(const Vector4& rectangle, const Vector2& p1, const Vector2& p2, const Vector2& p3);
	};
}