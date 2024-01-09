#pragma once

namespace Blueberry
{
	class BaseCamera;

	class SceneHelper
	{
	public:
		static Ray GUIPointToWorldRay(const Vector2& position, BaseCamera* camera);
	};
}