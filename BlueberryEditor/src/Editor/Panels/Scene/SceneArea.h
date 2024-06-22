#pragma once

#include "Blueberry\Math\Math.h"
#include "SceneCamera.h"
#include "SceneObjectPicker.h"

namespace Blueberry
{
	class RenderTexture;
	class Camera;

	class SceneArea
	{
	public:
		SceneArea();
		virtual ~SceneArea();
		
		void DrawUI();

		float GetPerspectiveDistance(const float objectSize, const float fov);
		float GetCameraDistance();

		BaseCamera* GetCamera();

		Vector3 GetPosition();
		void SetPosition(const Vector3& position);

		Quaternion GetRotation();
		void SetRotation(const Quaternion& rotation);

		float GetSize();
		void SetSize(const float& size);

		bool IsOrthographic();

		bool Is2DMode();
		void Set2DMode(const bool& is2DMode);

	private:
		Vector3 GetCameraPosition();
		Quaternion GetCameraRotation();

		Vector3 GetCameraTargetPosition();
		Quaternion GetCameraTargetRotation();
		float GetCameraOrthographicSize();

		void SetupCamera(const float& width, const float& height);
		void DrawControls();
		void DrawGizmos(const Rectangle& viewport);
		void DrawScene(const float width, const float height);
		void LookAt(const Vector3& point, const Quaternion& direction, const float& newSize, const bool& isOrthographic);

	private:
		RenderTexture* m_ColorMSAARenderTarget = nullptr;
		RenderTexture* m_DepthStencilMSAARenderTarget = nullptr;

		RenderTexture* m_ColorRenderTarget = nullptr;
		RenderTexture* m_DepthStencilRenderTarget = nullptr;
		
		Material* m_GridMaterial;
		Material* m_ResolveMSAAMaterial;
		SceneCamera m_Camera;
		SceneObjectPicker* m_ObjectPicker = nullptr;

		Vector3 m_Position = Vector3(0, 0, 0);
		Quaternion m_Rotation = Quaternion::Identity;
		Vector2 m_PreviousDragDelta = Vector2::Zero;
		// Radius of sphere camera is looking at
		float m_Size = 2;
		bool m_IsDragging = false;
		bool m_IsOrthographic = false;
		bool m_Is2DMode = false;

		float m_GizmoSnapping[3] = { 1.0f, 45.0f, 1.0f };
		int m_GizmoOperation = 7;

		Viewport m_Viewport;
	};
}