#pragma once

namespace Blueberry
{
	class Scene;
	class Camera;

	class GizmoRenderer
	{
	public:
		static void Draw(Scene* scene, Camera* camera);
	};
}