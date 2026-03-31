#pragma once

#include "Blueberry\Math\Math.h"
#include "SceneObjectPicker.h"

#include "Editor\Panels\EditorWindow.h"

namespace Blueberry
{
	class GfxTexture;
	class Camera;
	class ObjectUpdateEventArgs;

	class SceneArea : public EditorWindow
	{
		OBJECT_DECLARATION(SceneArea)

	public:
		SceneArea();
		virtual ~SceneArea();

		static void Open();
		
		virtual void OnDrawUI() final;
		virtual void OnSaveChanges() final;
		virtual void OnDiscardChanges() final;
		virtual String GetSaveChangesMessage() final;

		float GetPerspectiveDistance(float objectSize, float fov) const;
		float GetCameraDistance() const;

		Camera* GetCamera() const;

		const Vector3& GetPosition() const;
		void SetPosition(const Vector3& position);

		const Quaternion& GetRotation() const;
		void SetRotation(const Quaternion& rotation);

		float GetSize() const;
		void SetSize(float size);

		bool IsOrthographic() const;

		bool Is2DMode() const;
		void Set2DMode(bool is2DMode);

		static void RequestRedrawAll();

	private:
		Vector3 GetCameraPosition() const;
		Quaternion GetCameraRotation() const;

		Vector3 GetCameraTargetPosition() const;
		const Quaternion& GetCameraTargetRotation() const;
		float GetCameraOrthographicSize() const;

		void SetupCamera(float width, float height);
		void DrawControls();
		void DrawGizmos(const Rectangle& viewport);
		void DrawScene(float width, float height);
		void LookAt(const Vector3& point, const Quaternion& direction, float newSize, bool isOrthographic);

		void OnSelectionChange();
		void OnEntityUpdate();
		void OnObjectUpdate(const ObjectUpdateEventArgs& args);

	private:
		GfxTexture* m_ColorRenderTarget = nullptr;
		GfxTexture* m_DepthStencilRenderTarget = nullptr;
		GfxTexture* m_ColorCopyRenderTarget = nullptr;
		Material* m_GridMaterial;
		Camera* m_Camera;
		SceneObjectPicker* m_ObjectPicker = nullptr;

		Vector3 m_Position = Vector3(0, 0, 0);
		Quaternion m_Rotation = Quaternion::Identity;
		Vector2 m_PreviousDragDelta = Vector2::Zero;
		// Radius of sphere camera is looking at
		float m_Size = 2;
		bool m_IsDragging = false;
		bool m_IsOrthographic = false;
		bool m_Is2DMode = false;

		Vector3 m_PreviousPosition;
		Quaternion m_PreviousRotation;

		static uint32_t s_SceneRedrawRequestsCount;

		Viewport m_Viewport;
	};
}