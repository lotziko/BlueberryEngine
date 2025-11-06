#include "TransformEditor.h"

#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Camera.h"

#include "Editor\Misc\ImGuiHelper.h"
#include "Editor\Preferences.h"
#include "Editor\Panels\Scene\SceneArea.h"

#include <imgui\imgui.h>
#include <imgui\imguizmo.h>

namespace Blueberry
{
	void TransformEditor::OnEnable()
	{
		m_LocalPositionProperty = m_SerializedObject->FindProperty("m_LocalPosition");
		m_LocalRotationProperty = m_SerializedObject->FindProperty("m_LocalRotation");
		m_LocalScaleProperty = m_SerializedObject->FindProperty("m_LocalScale");
		m_LocalRotationEulerHintProperty = m_SerializedObject->FindProperty("m_LocalRotationEulerHint");
	}

	void TransformEditor::OnDrawInspector()
	{
		ImGui::Property(&m_LocalPositionProperty, "Position");
		if (ImGui::Property(&m_LocalRotationEulerHintProperty, "Rotation"))
		{
			Vector3 euler = m_LocalRotationEulerHintProperty.GetVector3();
			m_LocalRotationProperty.SetQuaternion(Quaternion::CreateFromYawPitchRoll(ToRadians(euler.y), ToRadians(euler.x), ToRadians(euler.z)));
		}
		ImGui::Property(&m_LocalScaleProperty, "Scale");

		if (m_SerializedObject->ApplyModifiedProperties())
		{
			SceneArea::RequestRedrawAll();
		}
	}

	// Based on https://discussions.unity.com/t/quaternion-to-three-hinge-joints/714182/10
	Vector3 CorrectEuler(Vector3 oldValue, Vector3 newValue)
	{
		float hint[3] = { ToRadians(oldValue.x), ToRadians(oldValue.y), ToRadians(oldValue.z) };
		float eul[3] = { ToRadians(newValue.x), ToRadians(newValue.y), ToRadians(newValue.z) };

		const float pi_thresh = 5.1f;
		const float pi_x2 = 2.0f * Pi;

		float dif[3] = {};

		for (uint32_t i = 0; i < 3; ++i)
		{
			dif[i] = eul[i] - hint[i];
			if (dif[i] > pi_thresh)
			{
				eul[i] -= floorf((dif[i] / pi_x2) + 0.5f) * pi_x2;
				dif[i] = eul[i] - hint[i];
			}
			else if (dif[i] < -pi_thresh)
			{
				eul[i] += floorf((-dif[i] / pi_x2) + 0.5f) * pi_x2;
				dif[i] = eul[i] - hint[i];
			}
		}

		if (fabs(dif[0]) > 3.2f && fabs(dif[1]) < 1.6f && fabs(dif[2]) < 1.6f)
		{
			if (dif[0] > 0.0f)
			{
				eul[0] -= pi_x2;
			}
			else
			{
				eul[0] += pi_x2;
			}
		}

		if (fabs(dif[1]) > 3.2f && fabs(dif[2]) < 1.6f && fabs(dif[0]) < 1.6f)
		{
			if (dif[1] > 0.0f)
			{
				eul[1] -= pi_x2;
			}
			else
			{
				eul[1] += pi_x2;
			}
		}

		if (fabs(dif[2]) > 3.2f && fabs(dif[0]) < 1.6f && fabs(dif[1]) < 1.6f)
		{
			if (dif[2] > 0.0f)
			{
				eul[2] -= pi_x2;
			}
			else
			{
				eul[2] += pi_x2;
			}
		}

		return ToDegrees(Vector3(eul[0], eul[1], eul[2]));
	}

	void TransformEditor::OnDrawSceneSelected()
	{
		// TODO update with SetLocalPosition() and targets instead
		Transform* transform = static_cast<Transform*>(m_SerializedObject->GetTarget());
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
				if (translation != m_LocalPositionProperty.GetVector3())
				{
					m_LocalPositionProperty.SetVector3(translation);
				}
				Quaternion previousRotation = m_LocalRotationProperty.GetQuaternion();
				if (rotation != previousRotation)
				{
					m_LocalRotationProperty.SetQuaternion(rotation);

					Vector3 previousEuler = m_LocalRotationEulerHintProperty.GetVector3();
					Vector3 euler = CorrectEuler(previousEuler, ToDegrees(rotation.ToEuler()));

					if (snapping[1] > 0)
					{
						euler /= snapping[1];
						euler = Vector3(roundf(euler.x), roundf(euler.y), roundf(euler.z));
						euler *= snapping[1];
					}

					if (fabs(euler.x) < 1e-6f)
					{
						euler.x = 0;
					}
					if (fabs(euler.y) < 1e-6f)
					{
						euler.y = 0;
					}
					if (fabs(euler.z) < 1e-6f)
					{
						euler.z = 0;
					}
					m_LocalRotationEulerHintProperty.SetVector3(euler);
				}
				if (scale != m_LocalScaleProperty.GetVector3())
				{
					m_LocalScaleProperty.SetVector3(scale);
				}
				m_SerializedObject->ApplyModifiedProperties();
				SceneArea::RequestRedrawAll();
			}
		}
	}
}