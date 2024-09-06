#pragma once

namespace Blueberry
{
	class Scene;
	class BaseCamera;

	class GizmoRenderer
	{
	public:
		static void Draw(Scene* scene, BaseCamera* camera);
	};
}