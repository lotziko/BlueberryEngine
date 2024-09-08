#include "bbpch.h"
#include "SceneHelper.h"
#include "Blueberry\Scene\Components\Camera.h"

namespace Blueberry
{
	Ray SceneHelper::GUIPointToWorldRay(const Vector2& position, Camera* camera)
	{
		// Based on https://github.com/Unity-Technologies/UnityCsReference/blob/master/Editor/Mono/Handles/HandleUtility.cs
		Vector2 viewportSize = camera->GetPixelSize();
		Matrix cameraToWorld = camera->GetViewMatrix().Invert();
		Matrix clipToCamera = camera->GetProjectionMatrix().Invert();
		float startZ = (camera->GetNearClipPlane());

		Vector3 rayOriginWorldSpace;
		Vector3 rayDirectionWorldSpace;

		Vector3 rayPointClipSpace = Vector3(position.x / viewportSize.x * 2 - 1, position.y / viewportSize.y * 2 - 1, 0.95f);
		Vector3 rayPointCameraSpace = MultiplyPoint(clipToCamera, rayPointClipSpace);

		if (camera->IsOrthographic())
		{
			Vector3 rayDirectionCameraSpace = Vector3(0.0f, 0.0f, -1.0f);
			rayDirectionWorldSpace = MultiplyVector(cameraToWorld, rayDirectionCameraSpace);
			rayDirectionWorldSpace.Normalize();

			Vector3 rayOriginCameraSpace = rayPointCameraSpace;
			// Do not invert startZ because of right handed coordinate system
			rayOriginCameraSpace.z = startZ;

			rayOriginWorldSpace = MultiplyPoint(cameraToWorld, rayOriginCameraSpace);
		}
		else
		{
			Vector3 rayDirectionCameraSpace = rayPointCameraSpace;
			rayDirectionCameraSpace.Normalize();

			rayDirectionWorldSpace = MultiplyVector(cameraToWorld, rayDirectionCameraSpace);

			Vector3 cameraPositionWorldSpace = MultiplyPoint(cameraToWorld, Vector3(0, 0, 0));
			// Do not invert startZ because of right handed coordinate system
			Vector3 originOffsetWorldSpace = rayDirectionWorldSpace * startZ / rayDirectionCameraSpace.z;
			rayOriginWorldSpace = cameraPositionWorldSpace + originOffsetWorldSpace;
		}

		return Ray(rayOriginWorldSpace, rayDirectionWorldSpace);
	}
}
