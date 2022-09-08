#pragma once

#include "Blueberry\Math\Math.h"
#include "SceneCamera.h"

namespace Blueberry
{
	class Scene;
	class Camera;

	class SceneArea
	{
	public:
		SceneArea() = default;
		SceneArea(const Ref<Scene>& scene);

		void Draw();
		void SetViewport(const Viewport& viewport);

	private:
		Ref<Scene> m_Scene;
		SceneCamera m_Camera;

		Viewport m_Viewport;
	};
}