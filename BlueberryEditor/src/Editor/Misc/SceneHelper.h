#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Camera;

	class SceneHelper
	{
	public:
		static Ray GUIPointToWorldRay(const Vector2& position, Camera* camera);
	};
}