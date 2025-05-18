#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class SceneArea;

	class SceneAreaMovement
	{
	public:
		static void HandleZoom(SceneArea* area, float delta, Vector2 mousePosition);
		static void HandleDrag(SceneArea* area, Vector2 delta);
	private:
		static Vector3 ScreenToWorldDistance(SceneArea* area, Vector2 delta);
	};
}