#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Camera;

	class CameraHelper
	{
	public:
		static RectangleFloat CalculateViewport(Camera* camera, const Rectangle& area);
	};
}