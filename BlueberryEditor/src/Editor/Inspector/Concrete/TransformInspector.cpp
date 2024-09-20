#include "bbpch.h"
#include "TransformInspector.h"

#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Camera.h"

#include "imgui\imgui.h"
#include "imgui\imguizmo.h"
#include "Editor\Misc\ImGuiHelper.h"

#include "Editor\Preferences.h"
#include "Editor\Panels\Scene\SceneArea.h"

namespace Blueberry
{
	void TransformInspector::Draw(Object* object)
	{
		Transform* transform = static_cast<Transform*>(object);
		bool hasChange = false;

		Vector3 localPosition = transform->GetLocalPosition();
		if (ImGui::DragVector3("Position", &localPosition))
		{
			transform->SetLocalPosition(localPosition);
			hasChange = true;
		}

		Vector3 localRotation = transform->GetLocalEulerRotationHint();
		if (ImGui::DragVector3("Rotation", &localRotation))
		{
			transform->SetLocalEulerRotationHint(localRotation);
			hasChange = true;
		}

		Vector3 localScale = transform->GetLocalScale();
		if (ImGui::DragVector3("Scale", &localScale))
		{
			transform->SetLocalScale(localScale);
			hasChange = true;
		}

		if (hasChange)
		{
			SceneArea::RequestRedrawAll();
		}
	}

	void TransformInspector::DrawScene(Object* object)
	{
		Transform* transform = static_cast<Transform*>(object);
		Matrix transformMatrix = Matrix::CreateScale(transform->GetLocalScale()) * Matrix::CreateFromQuaternion(transform->GetLocalRotation()) * Matrix::CreateTranslation(transform->GetLocalPosition());
		Transform* parentTransform = transform->GetParent();
		if (parentTransform != nullptr)
		{
			transformMatrix *= parentTransform->GetLocalToWorldMatrix();
		}
		
		Camera* camera = Camera::GetCurrent();
		if (camera != nullptr)
		{
			float snapping[3];
			float* gizmoSnapping = Preferences::GetGizmoSnapping();
			const int& gizmoOperation = Preferences::GetGizmoOperation();
			if (gizmoOperation == ImGuizmo::OPERATION::TRANSLATE)
			{
				snapping[0] = gizmoSnapping[0];
				snapping[1] = gizmoSnapping[0];
				snapping[2] = gizmoSnapping[0];
			}
			else if (gizmoOperation == ImGuizmo::OPERATION::ROTATE)
			{
				snapping[0] = gizmoSnapping[1];
				snapping[1] = gizmoSnapping[1];
				snapping[2] = gizmoSnapping[1];
			}
			else if (gizmoOperation == ImGuizmo::OPERATION::SCALE)
			{
				snapping[0] = gizmoSnapping[2];
				snapping[1] = gizmoSnapping[2];
				snapping[2] = gizmoSnapping[2];
			}

			if (ImGuizmo::Manipulate((float*)camera->GetViewMatrix().m, (float*)camera->GetProjectionMatrix().m, (ImGuizmo::OPERATION)gizmoOperation, ImGuizmo::MODE::LOCAL, (float*)transformMatrix.m, 0, snapping))
			{
				Vector3 scale;
				Quaternion rotation;
				Vector3 translation;

				if (parentTransform != nullptr)
				{
					transformMatrix *= parentTransform->GetLocalToWorldMatrix().Invert();
				}

				transformMatrix.Decompose(scale, rotation, translation);
				transform->SetLocalPosition(translation);
				transform->SetLocalRotationHint(rotation, gizmoSnapping[0]);
				transform->SetLocalScale(scale);
				SceneArea::RequestRedrawAll();
			}
		}
	}
}