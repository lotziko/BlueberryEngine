#pragma once

#include "Blueberry\Scene\EnityComponent.h"

namespace Blueberry
{
	class Camera : public Component
	{
		OBJECT_DECLARATION(Camera)
		COMPONENT_DECLARATION(Camera)
	public:
		Camera();
		~Camera() = default;

		void Update();

		const Matrix& GetProjectionMatrix() const { return m_ProjectionMatrix; }
		const Matrix& GetViewMatrix() const { return m_ViewMatrix; }

		const void SetResolution(const Vector2& resolution) { m_Resolution = resolution; }

	private:
		void RecalculateViewMatrix();
		void RecalculateProjectionMatrix();
	private:
		Matrix m_ProjectionMatrix;
		Matrix m_ViewMatrix;

		float m_PixelsPerUnit = 32;

		Vector2 m_Resolution = Vector2(480, 320);
		Vector3 m_Direction = Vector3(0, 0, -1);
		Vector3 m_Up = Vector3(0, 1, 0);

		float m_ZNearPlane = 1.0f;
		float m_ZFarPlane = 100.0f;
	};
}