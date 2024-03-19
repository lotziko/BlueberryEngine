#include "bbpch.h"
#include "BaseCamera.h"

namespace Blueberry
{
	const Matrix& BaseCamera::GetProjectionMatrix()
	{
		if (m_IsProjectionDirty)
		{
			RecalculateProjection();
			m_IsProjectionDirty = false;
		}
		return m_ProjectionMatrix;
	}

	const Matrix& BaseCamera::GetViewMatrix()
	{
		if (m_IsViewDirty)
		{
			RecalculateView();
			m_IsViewDirty = false;
		}
		return m_ViewMatrix;
	}

	const Matrix& BaseCamera::GetViewProjectionMatrix()
	{
		if (m_IsViewDirty)
		{
			RecalculateView();
			m_IsViewDirty = false;
		}
		if (m_IsProjectionDirty)
		{
			RecalculateProjection();
			m_IsProjectionDirty = false;
		}
		return m_ViewProjectionMatrix;
	}

	const Matrix& BaseCamera::GetInverseProjectionMatrix()
	{
		if (m_IsProjectionDirty)
		{
			RecalculateProjection();
			m_IsProjectionDirty = false;
		}
		return m_InverseProjectionMatrix;
	}

	const Matrix& BaseCamera::GetInverseViewMatrix()
	{
		if (m_IsViewDirty)
		{
			RecalculateView();
			m_IsViewDirty = false;
		}
		return m_InverseViewMatrix;
	}

	const Matrix& BaseCamera::GetInverseViewProjectionMatrix()
	{
		if (m_IsViewDirty)
		{
			RecalculateView();
			m_IsViewDirty = false;
		}
		if (m_IsProjectionDirty)
		{
			RecalculateProjection();
			m_IsProjectionDirty = false;
		}
		return m_InverseViewProjectionMatrix;
	}

	const bool& BaseCamera::IsOrthographic()
	{
		return m_IsOrthographic;
	}

	void BaseCamera::SetOrthographic(const bool& isOrthographic)
	{
		if (m_IsOrthographic != isOrthographic)
		{
			m_IsOrthographic = isOrthographic;
			m_IsProjectionDirty = true;
		}
	}

	const float& BaseCamera::GetOrthographicSize()
	{
		return m_OrthographicSize;
	}

	const void BaseCamera::SetOrthographicSize(const float& size)
	{
		m_OrthographicSize = size;
		m_IsProjectionDirty = true;
	}

	const Vector2& BaseCamera::GetPixelSize()
	{
		return m_PixelSize;
	}

	const void BaseCamera::SetPixelSize(const Vector2& pixelSize)
	{
		if (m_PixelSize != pixelSize)
		{
			m_PixelSize = pixelSize;
			m_AspectRatio = pixelSize.x / pixelSize.y;
			m_IsProjectionDirty = true;
		}
	}

	const float& BaseCamera::GetAspectRatio()
	{
		return m_AspectRatio;
	}

	void BaseCamera::SetAspectRatio(const float& aspectRatio)
	{
		if (m_AspectRatio != aspectRatio)
		{
			m_AspectRatio = aspectRatio;
			m_IsProjectionDirty = true;
		}
	}

	const float& BaseCamera::GetFieldOfView()
	{
		return m_FieldOfView;
	}

	void BaseCamera::SetFieldOfView(const float& fieldOfView)
	{
		if (m_FieldOfView != fieldOfView)
		{
			m_FieldOfView = fieldOfView;
			m_IsProjectionDirty = true;
		}
	}

	const Vector3& BaseCamera::GetPosition()
	{
		return m_Position;
	}

	void BaseCamera::SetPosition(const Vector3& position)
	{
		if (m_Position != position)
		{
			m_Position = position;
			m_IsViewDirty = true;
		}
	}

	const Quaternion& BaseCamera::GetRotation()
	{
		return m_Rotation;
	}

	void BaseCamera::SetRotation(const Quaternion& rotation)
	{
		if (m_Rotation != rotation)
		{
			m_Rotation = rotation;
			m_IsViewDirty = true;
		}
	}

	const float& BaseCamera::GetNearClipPlane()
	{
		return m_ZNearPlane;
	}

	void BaseCamera::SetNearClipPlane(const float& nearClip)
	{
		if (m_ZNearPlane != nearClip)
		{
			m_ZNearPlane = nearClip;
			m_IsProjectionDirty = true;
		}
	}

	const float& BaseCamera::GetFarClipPlane()
	{
		return m_ZFarPlane;
	}

	void BaseCamera::SetFarClipPlane(const float& farClip)
	{
		if (m_ZNearPlane != farClip)
		{
			m_ZFarPlane = farClip;
			m_IsProjectionDirty = true;
		}
	}

	BaseCamera* BaseCamera::GetCurrent()
	{
		return s_CurrentCamera;
	}

	void BaseCamera::SetCurrent(BaseCamera* camera)
	{
		s_CurrentCamera = camera;
	}

	Vector3 BaseCamera::WorldToScreenPoint(Vector3 position)
	{
		if (m_IsViewDirty)
		{
			RecalculateView();
			m_IsViewDirty = false;
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

	Vector3 BaseCamera::ScreenToWorldPoint(Vector3 position)
	{
		// Based on https://github.com/tezheng/UH5/blob/master/src/UnityEngine/Render/Camera.cs
		if (m_IsViewDirty)
		{
			RecalculateView();
			m_IsViewDirty = false;
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

	void BaseCamera::RecalculateView()
	{
		Matrix rotationMatrix = Matrix::CreateFromQuaternion(m_Rotation);

		Vector3 position = m_Position;
		Vector3 target = Vector3::Transform(m_Direction, rotationMatrix);
		target += position;

		Vector3 up = Vector3::Transform(m_Up, rotationMatrix);

		m_ViewMatrix = Matrix::CreateLookAt(position, target, up);
		m_InverseViewMatrix = m_ViewMatrix.Invert();
		m_ViewProjectionMatrix = m_ViewMatrix * m_ProjectionMatrix;
		m_InverseViewProjectionMatrix = m_ViewProjectionMatrix.Invert();
	}

	void BaseCamera::RecalculateProjection()
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
}