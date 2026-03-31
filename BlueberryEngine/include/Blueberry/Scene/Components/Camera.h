#pragma once

#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	enum class CameraType
	{
		Game,
		VR,
		Preview,
		Reflection,
	};

	class BB_API Camera : public Component
	{
		OBJECT_DECLARATION(Camera)

	public:
		Camera() = default;
		virtual ~Camera() = default;

		const Matrix& GetProjectionMatrix();
		const Matrix& GetViewMatrix();
		const Matrix& GetViewProjectionMatrix();
		const Matrix& GetInverseProjectionMatrix();
		const Matrix& GetInverseViewMatrix();
		const Matrix& GetInverseViewProjectionMatrix();

		bool IsOrthographic() const;
		void SetOrthographic(bool isOrthographic);

		float GetOrthographicSize() const;
		const void SetOrthographicSize(float size);

		const Vector2& GetPixelSize() const;
		const void SetPixelSize(const Vector2& pixelSize);

		float GetAspectRatio() const;
		void SetAspectRatio(float aspectRatio);

		float GetFieldOfView() const;
		void SetFieldOfView(float fieldOfView);

		float GetNearClipPlane() const;
		void SetNearClipPlane(float nearClip);

		float GetFarClipPlane() const;
		void SetFarClipPlane(float farClip);

		static Camera* GetCurrent();
		static void SetCurrent(Camera* camera);

		Vector3 WorldToScreenPoint(Vector3 position);
		Vector3 ScreenToWorldPoint(Vector3 position);

		CameraType GetCameraType() const;
		void SetCameraType(CameraType cameraType);

	private:
		bool IsViewDirty();
		void InvalidateProjection();

		void RecalculateView();
		void RecalculateProjection();

	private:
		inline static Camera* s_CurrentCamera = nullptr;

		Matrix m_ProjectionMatrix;
		Matrix m_ViewMatrix;
		Matrix m_InverseProjectionMatrix;
		Matrix m_InverseViewMatrix;
		Matrix m_ViewProjectionMatrix;
		Matrix m_InverseViewProjectionMatrix;

		bool m_IsProjectionDirty = true;

		bool m_IsOrthographic = true;
		float m_OrthographicSize = 5;
		Vector2 m_PixelSize = Vector2(480, 320);

		float m_FieldOfView = 60;
		float m_AspectRatio = 16.0f / 9.0f;

		Vector3 m_Direction = Vector3(0, 0, 1);
		Vector3 m_Up = Vector3(0, 1, 0);

		size_t m_UpdateCount = 0;

		float m_ZNearPlane = 0.1f;
		float m_ZFarPlane = 1000.0f;

		CameraType m_CameraType = CameraType::Game;

	private:
		Vector4 m_ShadowCascades[6];

		friend class RenderContext;
	};
}