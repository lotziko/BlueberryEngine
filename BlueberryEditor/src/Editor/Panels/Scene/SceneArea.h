#pragma once

#include "Blueberry\Math\Math.h"
#include "SceneCamera.h"

namespace Blueberry
{
	class Scene;
	class GfxTexture;
	class Camera;

	class SceneArea
	{
	public:
		SceneArea() = default;
		SceneArea(const Ref<Scene>& scene);
		
		void DrawUI();

		float GetCameraDistance();

	private:
		Vector3 GetCameraPosition();
		float GetCameraOrthographicSize();

		void DrawScene(const float width, const float height);

	private:
		Ref<Scene> m_Scene;
		Ref<GfxTexture> m_SceneRenderTarget;
		SceneCamera m_Camera;

		Vector3 m_Position = Vector3(0, 0, 0);
		Vector2 m_PreviousDragDelta = Vector2::Zero;
		// Radius of sphere camera is looking at
		float m_Size = 2;
		bool m_IsDragging = false;

		Viewport m_Viewport;

		friend class SceneAreaMovement;
	};
}