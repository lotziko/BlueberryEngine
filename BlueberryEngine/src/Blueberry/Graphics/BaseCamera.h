#pragma once

namespace Blueberry
{
	class BaseCamera
	{
	public:
		BaseCamera() = default;
		virtual ~BaseCamera() = default;

	public:
		const Matrix& GetProjectionMatrix();
		const Matrix& GetViewMatrix();
		const Matrix& GetViewProjectionMatrix();
		const Matrix& GetInverseProjectionMatrix();
		const Matrix& GetInverseViewMatrix();
		const Matrix& GetInverseViewProjectionMatrix();

		const bool& IsOrthographic();
		void SetOrthographic(const bool& isOrthographic);

		const float& GetOrthographicSize();
		const void SetOrthographicSize(const float& size);

		const Vector2& GetPixelSize();
		const void SetPixelSize(const Vector2& pixelSize);

		const float& GetAspectRatio();
		void SetAspectRatio(const float& aspectRatio);

		const float& GetFieldOfView();
		void SetFieldOfView(const float& fieldOfView);

		const Vector3& GetPosition();
		void SetPosition(const Vector3& position);

		const Quaternion& GetRotation();
		void SetRotation(const Quaternion& rotation);

		const float& GetNearClipPlane();
		void SetNearClipPlane(const float& nearClip);

		const float& GetFarClipPlane();
		void SetFarClipPlane(const float& farClip);

		Vector3 WorldToScreenPoint(Vector3 position);
		Vector3 ScreenToWorldPoint(Vector3 position);

	protected:
		void RecalculateView();
		void RecalculateProjection();

	protected:
		Matrix m_ProjectionMatrix;
		Matrix m_ViewMatrix;
		Matrix m_InverseProjectionMatrix;
		Matrix m_InverseViewMatrix;
		Matrix m_ViewProjectionMatrix;
		Matrix m_InverseViewProjectionMatrix;

		bool m_IsViewDirty = true;
		bool m_IsProjectionDirty = true;

		bool m_IsOrthographic = true;
		float m_OrthographicSize = 5;
		Vector2 m_PixelSize = Vector2(480, 320);
		
		float m_FieldOfView = 60;
		float m_AspectRatio = 16.0f / 9.0f;

		Vector3 m_Direction = Vector3(0, 0, -1);
		Vector3 m_Up = Vector3(0, 1, 0);

		Vector3 m_Position = Vector3::Zero;
		Quaternion m_Rotation = Quaternion::Identity;

		float m_ZNearPlane = 0.1f;
		float m_ZFarPlane = 1000.0f;
	};
}