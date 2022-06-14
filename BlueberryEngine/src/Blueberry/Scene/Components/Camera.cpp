#include "bbpch.h"
#include "Camera.h"

#include "Blueberry\Scene\EnityComponent.h"

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