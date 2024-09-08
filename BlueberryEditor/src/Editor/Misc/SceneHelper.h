#pragma once

namespace Blueberry
{
	class Camera;

	class SceneHelper
	{
	public:
		static Ray GUIPointToWorldRay(const Vector2& position, Camera* camera);
	};
}