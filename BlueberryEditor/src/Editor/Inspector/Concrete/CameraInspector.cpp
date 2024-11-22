#include "bbpch.h"
#include "CameraInspector.h"

#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Camera.h"

#include "imgui\imgui.h"
#include "Editor\Misc\ImGuiHelper.h"

#include "Editor\Gizmos\Gizmos.h"
#include "Editor\Panels\Scene\SceneArea.h"

#include "Blueberry\Graphics\RenderContext.h"

namespace Blueberry
{
	const char* CameraInspector::GetIconPath(Object* object)
	{
		return "assets/icons/CameraIcon.png";
	}

	void CameraInspector::Draw(Object* object)
	{
		Camera* camera = static_cast<Camera*>(object);
		bool hasChange = false;

		bool isOrthographic = camera->IsOrthographic();
		if (ImGui::BoolEdit("Is Orthographic", &isOrthographic))
		{
			camera->SetOrthographic(isOrthographic);
			hasChange = true;
		}

		if (isOrthographic)
		{
			Vector2 pixelSize = camera->GetPixelSize();
			if (ImGui::DragVector2("Pixel size", &pixelSize))
			{
				camera->SetPixelSize(pixelSize);
				hasChange = true;
			}
		}
		else
		{
			float fieldOfView = camera->GetFieldOfView();
			if (ImGui::SliderFloat("Field of View", &fieldOfView, 0.01f, 179.0f, "%1f", 0))
			{
				camera->SetFieldOfView(fieldOfView);
				hasChange = true;
			}
		}

		float nearClipPlane = camera->GetNearClipPlane();
		if (ImGui::FloatEdit("Near plane", &nearClipPlane))
		{
			camera->SetNearClipPlane(nearClipPlane);
			hasChange = true;
		}

		float farClipPlane = camera->GetFarClipPlane();
		if (ImGui::FloatEdit("Far plane", &farClipPlane))
		{
			camera->SetFarClipPlane(farClipPlane);
			hasChange = true;
		}

		if (hasChange)
		{
			SceneArea::RequestRedrawAll();
		}
	}

	void CameraInspector::DrawScene(Object* object)
	{
		Camera* camera = static_cast<Camera*>(object);
		Transform* transform = camera->GetTransform();

		Matrix view = camera->GetInverseViewMatrix();
		Matrix projection = camera->GetProjectionMatrix();
		Frustum frustum;
		frustum.CreateFromMatrix(frustum, projection, false);
		frustum.Transform(frustum, view);

		Gizmos::SetMatrix(Matrix::Identity);
		Gizmos::DrawFrustum(frustum);
	}
}