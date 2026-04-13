#include "Blueberry\Tools\CameraHelper.h"

#include "Blueberry\Scene\Components\Camera.h"

namespace Blueberry
{
	RectangleFloat CameraHelper::CalculateViewport(Camera* camera, const Rectangle& area)
	{
		Vector2 pos = Vector2(static_cast<float>(area.x), static_cast<float>(area.y));
		Vector2 size = Vector2(static_cast<float>(area.width), static_cast<float>(area.height));
		float areaAspectRatio = size.x / size.y;
		float cameraAspectRatio = camera->GetAspectRatio();

		RectangleFloat result = {};
		if (areaAspectRatio > cameraAspectRatio)
		{
			result.width = size.y * cameraAspectRatio;
			result.x = pos.x + (size.x - result.width) / 2.0f;
			result.y = pos.y;
			result.height = size.y;
		}
		else
		{
			result.height = size.x / cameraAspectRatio;
			result.y = pos.y + (size.y - result.height) / 2.0f;
			result.x = pos.x;
			result.width = size.x;
		}
		return result;
	}
}