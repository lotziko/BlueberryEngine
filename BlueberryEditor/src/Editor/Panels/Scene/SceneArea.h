#pragma once

#include "Blueberry\Math\Math.h"
#include "SceneCamera.h"
#include "SceneObjectPicker.h"

namespace Blueberry
{
	class GfxTexture;
	class Camera;

	class SceneArea
	{
	public:
		SceneArea();
		
		void DrawUI();

		float GetCameraDistance();

	private:
		Vector3 GetCameraPosition();
		float GetCameraOrthographicSize();

		void DrawScene(const float width, const float height);

	private:
		GfxTexture* m_SceneRenderTarget;
		SceneCamera m_Camera;
		SceneObjectPicker m_ObjectPicker;

		Vector3 m_Position = Vector3(0, 0, 0);
		Vector2 m_PreviousDragDelta = Vector2::Zero;
		// Radius of sphere camera is looking at
		float m_Size = 2;
		bool m_IsDragging = false;

		Viewport m_Viewport;

		friend class SceneAreaMovement;
	};
}