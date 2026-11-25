#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Camera;

	class SceneHelper
	{
	public:
		static Ray ScreenPointToWorldRay(const Vector2& position, Camera* camera);
	};
}