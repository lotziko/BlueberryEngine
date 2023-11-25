#include "bbpch.h"
#include "SceneAreaMovement.h"

#include "SceneArea.h"

namespace Blueberry
{
	Ray GUIPointToWorldRay(Vector2 position, SceneCamera* camera)
	{
		Vector2 pixelSize = camera->GetPixelSize();
		Matrix inverseView = camera->GetViewMatrix().Invert();
		Matrix inverseProjection = camera->GetProjectionMatrix().Invert();

		Vector4 rayOriginWorldSpace;
		Vector4 rayDirectionWorldSpace;

		Vector4 origin = Vector4(position.x / pixelSize.x * 2 - 1, position.y / pixelSize.y * 2 - 1, 0.95f, 1.0f);
		Vector4 projectedOrigin = Vector4::Transform(origin, inverseProjection);
		projectedOrigin.w = 1 / projectedOrigin.w;
		projectedOrigin.x *= projectedOrigin.w;
		projectedOrigin.y *= projectedOrigin.w;
		projectedOrigin.z *= projectedOrigin.w;

		if (true)
		{
			Vector4 rayDirectionViewSpace = Vector4(0.0f, 0.0f, -1.0f, 0.0f);
			rayDirectionWorldSpace = Vector4::Transform(rayDirectionViewSpace, inverseView);
			rayDirectionWorldSpace.Normalize();

			Vector4 rayOriginViewSpace = projectedOrigin;
			rayOriginViewSpace.z = (camera->GetNearClipPlane());
			rayOriginWorldSpace = Vector4::Transform(rayOriginViewSpace, inverseView);
			rayOriginWorldSpace.w = 1 / rayOriginWorldSpace.w;
			rayOriginWorldSpace.x *= rayOriginWorldSpace.w;
			rayOriginWorldSpace.y *= rayOriginWorldSpace.w;
			rayOriginWorldSpace.z *= rayOriginWorldSpace.w;
		}

		return Ray(Vector3(rayOriginWorldSpace.x, rayOriginWorldSpace.y, rayOriginWorldSpace.z), Vector3(rayDirectionWorldSpace.x, rayDirectionWorldSpace.y, rayDirectionWorldSpace.z));
	}

	void SceneAreaMovement::Test(SceneArea* area, Vector2 mousePosition)
	{
		auto ray = GUIPointToWorldRay(mousePosition, &(area->m_Camera));
		//BB_INFO(std::string() << ray.position.x << " " << ray.position.y << " " << ray.position.z << "     " << ray.direction.x << " " << ray.direction.y << " " << ray.direction.z);
	}

	void SceneAreaMovement::HandleZoom(SceneArea* area, float delta, Vector2 mousePosition)
	{
		float targetSize;

		if (area->m_Camera.IsOrthographic())
		{
			targetSize = abs(area->m_Size) * (-delta * 0.015f + 1.0f);
		}
		else
		{
			// TODO Perspective
			targetSize = 1;
		}

		float initialDistance = area->GetCameraDistance();
		area->m_Size = targetSize;

		float percentage = 1.0f - (area->GetCameraDistance() / initialDistance);

		Ray mouseRay = GUIPointToWorldRay(mousePosition, &(area->m_Camera));
		Vector3 mousePivot = mouseRay.position + mouseRay.direction * initialDistance;
		Vector3 pivotVector = mousePivot - area->m_Position;
		
		area->m_Position += pivotVector * percentage;
	}

	Vector2 GetDynamicClipPlanes(float size)
	{
		const float k_MaxCameraFarClip = 1.844674E+19f;
		float farClip = Min(Max(2000.0f * size, 1000.0f), k_MaxCameraFarClip);
		return Vector2(farClip * 0.000005f, farClip);
	}

	void SceneAreaMovement::HandleDrag(SceneArea* area, Vector2 delta)
	{
		const float k_MaxCameraSizeForWorldToScreen = 2.5E+7f;

		Vector2 screenDelta = Vector2(-delta.x, delta.y);

		SceneCamera* camera = &(area->m_Camera);
		Vector3 previousPosition = camera->GetPosition();
		float previousNear = camera->GetNearClipPlane();
		float previousFar = camera->GetFarClipPlane();
		float size = area->m_Size;

		area->m_Size = Min(size, k_MaxCameraSizeForWorldToScreen);
		float scale = size / area->m_Size;
		Vector2 clip = GetDynamicClipPlanes(area->m_Size);

		camera->SetNearClipPlane(clip.x);
		camera->SetFarClipPlane(clip.y);
		camera->SetPosition(Vector3::Zero);

		Vector3 position = Vector3::Transform(Vector3(0, 0, area->GetCameraDistance()), camera->GetRotation());
		Vector3 screenPosition = camera->WorldToScreenPoint(position);
		screenPosition += Vector3(screenDelta.x, screenDelta.y, 0);
		Vector3 worldDelta = camera->ScreenToWorldPoint(screenPosition) - position;
		worldDelta *= scale;

		camera->SetPosition(previousPosition);
		camera->SetNearClipPlane(previousNear);
		camera->SetFarClipPlane(previousFar);

		area->m_Position += worldDelta;
	}
}
