#include "CameraInspector.h"

#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Camera.h"

#include "Editor\Misc\ImGuiHelper.h"

#include "Editor\Gizmos\Gizmos.h"
#include "Editor\Panels\Scene\SceneArea.h"

#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\Texture2D.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	static Texture* s_Icon = nullptr;

	CameraInspector::CameraInspector()
	{
		if (s_Icon == nullptr)
		{
			s_Icon = static_cast<Texture*>(AssetLoader::Load("assets/icons/CameraIcon.png"));
		}
	}

	Texture* CameraInspector::GetIcon(Object* object)
	{
		return s_Icon;
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
			float orthographicSize = camera->GetOrthographicSize();
			if (ImGui::FloatEdit("Orthographic size", &orthographicSize))
			{
				camera->SetOrthographicSize(orthographicSize);
				hasChange = true;
			}

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
			if (ImGui::FloatEdit("Field of View", &fieldOfView, 0.01f, 179.0f))
			{
				camera->SetFieldOfView(fieldOfView);
				hasChange = true;
			}
		}

		float nearClipPlane = camera->GetNearClipPlane();
		if (ImGui::FloatEdit("Near plane", &nearClipPlane, 0.01f))
		{
			camera->SetNearClipPlane(std::max(nearClipPlane, 0.01f));
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