#include "bbpch.h"
#include "Camera.h"

#include "Blueberry\Scene\EnityComponent.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Component, Camera)

	const Matrix& Camera::GetProjectionMatrix()
	{
		if (m_IsProjectionDirty)
		{
			RecalculateProjection();
			m_IsProjectionDirty = false;
		}
		return m_ProjectionMatrix;
	}

	const Matrix& Camera::GetViewMatrix()
	{
		if (IsViewDirty())
		{
			RecalculateView();
		}
		return m_ViewMatrix;
	}

	const Matrix& Camera::GetViewProjectionMatrix()
	{
		if (IsViewDirty())
		{
			RecalculateView();
		}
		if (m_IsProjectionDirty)
		{
			RecalculateProjection();
			m_IsProjectionDirty = false;
		}
		return m_ViewProjectionMatrix;
	}

	const Matrix& Camera::GetInverseProjectionMatrix()
	{
		if (m_IsProjectionDirty)
		{
			RecalculateProjection();
			m_IsProjectionDirty = false;
		}
		return m_InverseProjectionMatrix;
	}

	const Matrix& Camera::GetInverseViewMatrix()
	{
		if (IsViewDirty())
		{
			RecalculateView();
		}
		return m_InverseViewMatrix;
	}

	const Matrix& Camera::GetInverseViewProjectionMatrix()
	{
		if (IsViewDirty())
		{
			RecalculateView();
		}
		if (m_IsProjectionDirty)
		{
			RecalculateProjection();
			m_IsProjectionDirty = false;
		}
		return m_InverseViewProjectionMatrix;
	}

	const bool& Camera::IsOrthographic()
	{
		return m_IsOrthographic;
	}

	void Camera::SetOrthographic(const bool& isOrthographic)
	{
		if (m_IsOrthographic != isOrthographic)
		{
			m_IsOrthographic = isOrthographic;
			m_IsProjectionDirty = true;
		}
	}

	const float& Camera::GetOrthographicSize()
	{
		return m_OrthographicSize;
	}

	const void Camera::SetOrthographicSize(const float& size)
	{
		m_OrthographicSize = size;
		m_IsProjectionDirty = true;
	}

	const Vector2& Camera::GetPixelSize()
	{
		return m_PixelSize;
	}

	const void Camera::SetPixelSize(const Vector2& pixelSize)
	{
		if (m_PixelSize != pixelSize)
		{
			m_PixelSize = pixelSize;
			m_AspectRatio = pixelSize.x / pixelSize.y;
			m_IsProjectionDirty = true;
		}
	}

	const float& Camera::GetAspectRatio()
	{
		return m_AspectRatio;
	}

	void Camera::SetAspectRatio(const float& aspectRatio)
	{
		if (m_AspectRatio != aspectRatio)
		{
			m_AspectRatio = aspectRatio;
			m_IsProjectionDirty = true;
		}
	}

	const float& Camera::GetFieldOfView()
	{
		return m_FieldOfView;
	}

	void Camera::SetFieldOfView(const float& fieldOfView)
	{
		if (m_FieldOfView != fieldOfView)
		{
			m_FieldOfView = fieldOfView;
			m_IsProjectionDirty = true;
		}
	}

	const float& Camera::GetNearClipPlane()
	{
		return m_ZNearPlane;
	}

	void Camera::SetNearClipPlane(const float& nearClip)
	{
		if (m_ZNearPlane != nearClip)
		{
			m_ZNearPlane = nearClip;
			m_IsProjectionDirty = true;
		}
	}

	const float& Camera::GetFarClipPlane()
	{
		return m_ZFarPlane;
	}

	void Camera::SetFarClipPlane(const float& farClip)
	{
		if (m_ZNearPlane != farClip)
		{
			m_ZFarPlane = farClip;
			m_IsProjectionDirty = true;
		}
	}

	Camera* Camera::GetCurrent()
	{
		return s_CurrentCamera;
	}

	void Camera::SetCurrent(Camera* camera)
	{
		s_CurrentCamera = camera;
	}

	Vector3 Camera::WorldToScreenPoint(Vector3 position)
	{
		if (IsViewDirty())
		{
			RecalculateView();
		}
		if (m_IsProjectionDirty)
		{
			RecalculateProjection();
			m_IsProjectionDirty = false;
		}

		Vector4 screenPosition = Vector4::Transform(Vector4(position.x, position.y, position.z, 1), m_ViewProjectionMatrix);

		if (screenPosition.w == 0)
		{
			return Vector3::Zero;
		}
		else
		{
			Vector3 cameraPosition = m_InverseViewMatrix.Translation();
			Vector3 direction = position - cameraPosition;
			// Use backward to match right handed coordinate system
			Vector3 forward = -m_InverseViewMatrix.Forward();
			float distance = direction.Dot(forward);

			screenPosition.x = (screenPosition.x / screenPosition.w + 1) * 0.5f * m_PixelSize.x;
			screenPosition.y = (screenPosition.y / screenPosition.w + 1) * 0.5f * m_PixelSize.y;
			return Vector3(screenPosition.x, screenPosition.y, distance);
		}
	}

	Vector3 Camera::ScreenToWorldPoint(Vector3 position)
	{
		// Based on https://github.com/tezheng/UH5/blob/master/src/UnityEngine/Render/Camera.cs
		if (IsViewDirty())
		{
			RecalculateView();
		}
		if (m_IsProjectionDirty)
		{
			RecalculateProjection();
			m_IsProjectionDirty = false;
		}

		Vector4 screenPosition = Vector4(2 * (position.x / m_PixelSize.x) - 1, 2 * (position.y / m_PixelSize.y) - 1, 0.95f, 1.0f);
		Vector3 worldPosition = Vector3::Transform(Vector3(screenPosition.x, screenPosition.y, screenPosition.z), m_InverseViewProjectionMatrix);

		Vector3 cameraPosition = m_InverseViewMatrix.Translation();
		Vector3 direction = worldPosition - cameraPosition;
		// Use backward to match right handed coordinate system
		Vector3 forward = -m_InverseViewMatrix.Forward();
		float distance = direction.Dot(forward);

		if (abs(distance) > 1.0e-6f)
		{
			if (m_IsOrthographic)
			{
				return worldPosition - forward * (distance - position.z);
			}
			else
			{
				direction *= position.z / distance;
				return cameraPosition + direction;
			}
		}
		return Vector3::Zero;
	}

	bool Camera::IsViewDirty()
	{
		// This may cause problems some time later
		size_t transformRecalculationFrame = GetEntity()->GetTransform()->GetRecalculationFrame();
		if (m_RecalculationFrame <= transformRecalculationFrame)
		{
			m_RecalculationFrame = transformRecalculationFrame;
			return true;
		}
		return false;
	}

	void Camera::RecalculateView()
	{
		Transform* transform = GetEntity()->GetTransform();
		Matrix rotationMatrix = Matrix::CreateFromQuaternion(transform->GetRotation());

		Vector3 position = transform->GetPosition();
		Vector3 target = Vector3::Transform(m_Direction, rotationMatrix);
		target += position;

		Vector3 up = Vector3::Transform(m_Up, rotationMatrix);

		m_ViewMatrix = Matrix::CreateLookAt(position, target, up);
		m_InverseViewMatrix = m_ViewMatrix.Invert();
		m_ViewProjectionMatrix = m_ViewMatrix * m_ProjectionMatrix;
		m_InverseViewProjectionMatrix = m_ViewProjectionMatrix.Invert();
	}

	void Camera::RecalculateProjection()
	{
		if (m_IsOrthographic)
		{
			m_ProjectionMatrix = Matrix::CreateOrthographic((m_OrthographicSize * 2 * m_AspectRatio), (m_OrthographicSize * 2), m_ZNearPlane, m_ZFarPlane);
		}
		else
		{
			m_ProjectionMatrix = Matrix::CreatePerspectiveFieldOfView(ToRadians(m_FieldOfView), m_AspectRatio, m_ZNearPlane, m_ZFarPlane);
		}
		m_InverseProjectionMatrix = m_ProjectionMatrix.Invert();
		m_ViewProjectionMatrix = m_ViewMatrix * m_ProjectionMatrix;
		m_InverseViewProjectionMatrix = m_ViewProjectionMatrix.Invert();
	}

	void Camera::BindProperties()
	{
		BEGIN_OBJECT_BINDING(Camera)
		BIND_FIELD(FieldInfo(TO_STRING(m_Entity), &Camera::m_Entity, BindingType::ObjectPtr).SetObjectType(Entity::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_IsOrthographic), &Camera::m_IsOrthographic, BindingType::Bool))
		BIND_FIELD(FieldInfo(TO_STRING(m_OrthographicSize), &Camera::m_OrthographicSize, BindingType::Float))
		BIND_FIELD(FieldInfo(TO_STRING(m_PixelSize), &Camera::m_PixelSize, BindingType::Vector2))
		BIND_FIELD(FieldInfo(TO_STRING(m_AspectRatio), &Camera::m_FieldOfView, BindingType::Float))
		BIND_FIELD(FieldInfo(TO_STRING(m_FieldOfView), &Camera::m_FieldOfView, BindingType::Float))
		BIND_FIELD(FieldInfo(TO_STRING(m_AspectRatio), &Camera::m_AspectRatio, BindingType::Float))
		BIND_FIELD(FieldInfo(TO_STRING(m_ZNearPlane), &Camera::m_ZNearPlane, BindingType::Float))
		BIND_FIELD(FieldInfo(TO_STRING(m_ZFarPlane), &Camera::m_ZFarPlane, BindingType::Float))
		END_OBJECT_BINDING()
	}
}