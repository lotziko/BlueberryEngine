#pragma once

namespace Blueberry
{
	class SceneArea;

	class SceneAreaMovement
	{
	public:
		static void Test(SceneArea* area, Vector2 mousePosition);
		static void HandleZoom(SceneArea* area, float delta, Vector2 mousePosition);
		static void HandleDrag(SceneArea* area, Vector2 delta);
	};
}