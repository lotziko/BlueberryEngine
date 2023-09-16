#include "bbpch.h"
#include "Camera.h"

#include "Blueberry\Scene\EnityComponent.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Component, Camera)

	Camera::Camera()
	{
		RecalculateProjectionMatrix();
	}

	void Camera::Update()
	{
		RecalculateViewMatrix();
		RecalculateProjectionMatrix();
	}

	const Matrix& Camera::GetProjectionMatrix() const
	{
		return m_ProjectionMatrix;
	}

	const Matrix& Camera::GetViewMatrix() const
	{
		return m_ViewMatrix;
	}

	const void Camera::SetResolution(const Vector2& resolution)
	{
		m_Resolution = resolution;
	}

	std::string Camera::ToString() const
	{
		return "Camera";
	}

	void Camera::RecalculateViewMatrix()
	{
		Transform* transform = GetEntity()->GetTransform();
		Vector3 position = transform->GetLocalPosition();
		Quaternion rotation = transform->GetLocalRotation();

		Matrix rotationMatrix = Matrix::CreateFromQuaternion(rotation);
		Vector3 target = Vector3::Transform(m_Direction, rotationMatrix);
		target += position;

		Vector3 up = Vector3::Transform(m_Up, rotationMatrix);

		m_ViewMatrix = Matrix::CreateLookAt(position, target, up);
	}

	void Camera::RecalculateProjectionMatrix()
	{
		m_ProjectionMatrix = Matrix::CreateOrthographic(m_Resolution.x / m_PixelsPerUnit, m_Resolution.y / m_PixelsPerUnit, m_ZNearPlane, m_ZFarPlane);
	}
}