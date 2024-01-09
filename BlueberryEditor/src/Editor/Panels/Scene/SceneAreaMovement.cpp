#include "bbpch.h"
#include "SceneAreaMovement.h"
#include "SceneArea.h"
#include "Editor\Misc\SceneHelper.h"

namespace Blueberry
{
	void SceneAreaMovement::HandleZoom(SceneArea* area, float delta, Vector2 mousePosition)
	{
		bool zoomTowardsCenter = !area->Is2DMode();
		float targetSize;

		if (area->IsOrthographic())
		{
			targetSize = abs(area->GetSize()) * (-delta * 0.015f + 1.0f);
		}
		else
		{
			float relativeDelta = abs(area->GetSize()) * -delta * 0.015f;
			const float minDelta = 0.0001f;
			if (relativeDelta > 0 && relativeDelta < minDelta)
			{
				relativeDelta = minDelta;
			}
			else if (relativeDelta < 0 && relativeDelta > -minDelta)
			{
				relativeDelta = -minDelta;
			}
			targetSize = area->GetSize() + relativeDelta;
		}

		float initialDistance = area->GetCameraDistance();
	
		if (!(isnan(targetSize) || isinf(targetSize)))
		{
			const float maxSceneViewSize = 3.2e34f;
			area->SetSize(Min(targetSize, maxSceneViewSize));
		}

		if (!zoomTowardsCenter && abs(area->GetCameraDistance() < 1.0e7f))
		{
			float percentage = 1.0f - (area->GetCameraDistance() / initialDistance);

			Ray mouseRay = SceneHelper::GUIPointToWorldRay(mousePosition, area->GetCamera());
			Vector3 mousePivot = mouseRay.position + mouseRay.direction * initialDistance;
			Vector3 pivotVector = mousePivot - area->GetPosition();

			area->SetPosition(area->GetPosition() + pivotVector * percentage);
		}
	}

	Vector2 GetDynamicClipPlanes(float size)
	{
		const float maxCameraFarClip = 1.844674E+19f;

		float farClip = Min(Max(2000.0f * size, 1000.0f), maxCameraFarClip);
		return Vector2(farClip * 0.000005f, farClip);
	}

	void SceneAreaMovement::HandleDrag(SceneArea* area, Vector2 delta)
	{
		if (area->Is2DMode())
		{
			// 2D pan
			Vector3 worldDelta = ScreenToWorldDistance(area, Vector2(-delta.x, delta.y));
			area->SetPosition(area->GetPosition() + worldDelta);
		}
		else
		{
			if (area->IsOrthographic())
			{

			}
			else
			{
				// FPS
				Vector3 cameraPosition = area->GetPosition() - Vector3::Transform(Vector3::Forward, area->GetRotation()) * area->GetCameraDistance();
				Quaternion rotation = area->GetRotation();
				// Right handed coordinate system requires inverted delta and a different order of quaternion multiplication
				rotation *= Quaternion::CreateFromAxisAngle(Vector3::Transform(Vector3::Right, rotation), -delta.y * 0.003f);
				rotation *= Quaternion::CreateFromAxisAngle(Vector3::Up, -delta.x * 0.003f);
				area->SetRotation(rotation);
				area->SetPosition(cameraPosition + Vector3::Transform(Vector3::Forward, rotation) * area->GetCameraDistance());
			}
		}
	}

	Vector3 SceneAreaMovement::ScreenToWorldDistance(SceneArea* area, Vector2 delta)
	{
		const float maxCameraSizeForWorldToScreen = 2.5E+7f;

		BaseCamera* camera = area->GetCamera();
		Vector3 previousPosition = camera->GetPosition();
		float previousNear = camera->GetNearClipPlane();
		float previousFar = camera->GetFarClipPlane();
		float size = area->GetSize();

		area->SetSize(Min(size, maxCameraSizeForWorldToScreen));
		float scale = size / area->GetSize();
		Vector2 clip = GetDynamicClipPlanes(area->GetSize());

		camera->SetNearClipPlane(clip.x);
		camera->SetFarClipPlane(clip.y);
		camera->SetPosition(Vector3::Zero);

		Vector3 position = Vector3::Transform(Vector3(0, 0, area->GetCameraDistance()), camera->GetRotation());
		Vector3 screenPosition = camera->WorldToScreenPoint(position);
		screenPosition += Vector3(delta.x, delta.y, 0);
		Vector3 worldDelta = camera->ScreenToWorldPoint(screenPosition) - position;
		worldDelta *= scale;

		camera->SetPosition(previousPosition);
		camera->SetNearClipPlane(previousNear);
		camera->SetFarClipPlane(previousFar);

		return worldDelta;
	}
}
