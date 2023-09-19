#pragma once

#include "Blueberry\Math\Math.h"
#include "SceneCamera.h"

namespace Blueberry
{
	class Scene;
	class Texture;
	class Camera;

	class SceneArea
	{
	public:
		SceneArea() = default;
		SceneArea(const Ref<Scene>& scene);
		
		void DrawUI();

	private:
		void DrawScene(const float width, const float height);

	private:
		Ref<Scene> m_Scene;
		Ref<Texture> m_SceneRenderTarget;
		SceneCamera m_Camera;

		Viewport m_Viewport;
	};
}