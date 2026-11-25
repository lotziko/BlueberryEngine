#include "TransformEditor.h"

#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Camera.h"

#include "Editor\Misc\ImGuiHelper.h"
#include "Editor\Gizmos\Gizmos.h"
#include "Editor\Preferences.h"
#include "Editor\Panels\Scene\SceneArea.h"
#include "Editor\Panels\Hierarchy\SceneHierarchy.h"
#include "Editor\Undo\UndoManager.h"

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
		m_IsStaticProperty = m_SerializedObject->FindProperty("m_IsStatic");
		SceneHierarchy::GetHierarchyUpdated().AddCallback<TransformEditor, &TransformEditor::OnHierarchyUpdate>(this);
		ImGui::Events::GetClearedOverride().AddCallback<&TransformEditor::OnPropertyClear>();
	}

	void TransformEditor::OnDisable()
	{
		SceneHierarchy::GetHierarchyUpdated().RemoveCallback<TransformEditor, &TransformEditor::OnHierarchyUpdate>(this);
		ImGui::Events::GetClearedOverride().RemoveCallback<&TransformEditor::OnPropertyClear>();
	}

	void TransformEditor::OnDrawInspector()
	{
		m_SerializedObject->UpdateIfNeeded();

		ImGui::Property(&m_LocalPositionProperty, "Position");
		if (ImGui::Property(&m_LocalRotationEulerHintProperty, "Rotation"))
		{
			Vector3 euler = m_LocalRotationEulerHintProperty.GetVector3();
			m_LocalRotationProperty.SetQuaternion(Quaternion::CreateFromYawPitchRoll(Math::Math::ToRadians(euler.y), Math::Math::ToRadians(euler.x), Math::Math::ToRadians(euler.z)));
		}
		ImGui::Property(&m_LocalScaleProperty, "Scale");
		ImGui::Property(&m_IsStaticProperty, "Is Static");

		m_SerializedObject->ApplyModifiedProperties();
	}

	// Based on https://discussions.unity.com/t/quaternion-to-three-hinge-joints/714182/10
	Vector3 CorrectEuler(Vector3 oldValue, Vector3 newValue)
	{
		float hint[3] = { Math::ToRadians(oldValue.x), Math::ToRadians(oldValue.y), Math::ToRadians(oldValue.z) };
		float eul[3] = { Math::ToRadians(newValue.x), Math::ToRadians(newValue.y), Math::ToRadians(newValue.z) };

		const float pi_thresh = 5.1f;
		const float pi_x2 = 2.0f * Math::Pi;

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

		return Math::ToDegrees(Vector3(eul[0], eul[1], eul[2]));
	}

	float WrapRadians(float angle)
	{
		return atan2f(sinf(angle), cosf(angle));
	}

	Vector3 CorrectEuler(const Vector3& previousEuler, const Quaternion& rotation)
	{
		Vector3 principal = rotation.ToEuler();
		Vector3 candidateA = Math::ToDegrees(principal);
		Vector3 candidateB = candidateA;

		if (fabsf(cosf(principal.x)) > 1e-6f)
		{
			Vector3 alternate(principal.x >= 0.0f ? Math::Pi - principal.x : -Math::Pi - principal.x, WrapRadians(principal.y + Math::Pi), WrapRadians(principal.z + Math::Pi));
			candidateB = Math::ToDegrees(alternate);
		}

		candidateA = CorrectEuler(previousEuler, candidateA);
		candidateB = CorrectEuler(previousEuler, candidateB);

		float distanceA = fabsf(candidateA.x - previousEuler.x) + fabsf(candidateA.y - previousEuler.y) + fabsf(candidateA.z - previousEuler.z);
		float distanceB = fabsf(candidateB.x - previousEuler.x) + fabsf(candidateB.y - previousEuler.y) + fabsf(candidateB.z - previousEuler.z);
		
		return distanceA <= distanceB ? candidateA : candidateB;
	}

	void TransformEditor::OnDrawSceneSelected()
	{
		int gizmoOperation = Preferences::GetGizmoOperation();
		bool isGizmoLocal = Preferences::IsGizmoLocalSpace();
		Transform* transform = static_cast<Transform*>(m_SerializedObject->GetTarget());

		Transform* parentTransform = transform->GetParent();
		Matrix transformMatrix = Matrix::Identity;
		if (parentTransform != nullptr)
		{
			transformMatrix = parentTransform->GetLocalToWorldMatrix();
		}
		Gizmos::SetMatrix(transformMatrix);

		Vector3 position = transform->GetLocalPosition();
		Quaternion rotation = transform->GetLocalRotation();
		Vector3 scale = transform->GetLocalScale();

		switch (gizmoOperation)
		{
		case 0:
		{
			if (Gizmos::PositionHandle(position, rotation, isGizmoLocal))
			{
				Vector3 delta = position - transform->GetLocalPosition();
				Vector3 worldDelta = Math::MultiplyVector(transformMatrix, delta);

				for (Object* target : m_SerializedObject->GetTargets())
				{
					Transform* targetTransform = static_cast<Transform*>(target);
					UndoManager::RecordObject(targetTransform);
					targetTransform->SetPosition(targetTransform->GetPosition() + worldDelta);
				}
			}
		}
		break;
		case 1:
		{
			if (Gizmos::RotationHandle(position, rotation, isGizmoLocal))
			{
				Quaternion inverseRotation;
				transform->GetLocalRotation().Inverse(inverseRotation);
				Quaternion delta = inverseRotation * rotation;
				for (Object* target : m_SerializedObject->GetTargets())
				{
					Transform* targetTransform = static_cast<Transform*>(target);
					UndoManager::RecordObject(targetTransform);
					targetTransform->SetLocalRotation(targetTransform->GetLocalRotation() * delta, false);
					Vector3 previousEuler = transform->GetLocalEulerRotationHint();
					Vector3 euler = CorrectEuler(previousEuler, rotation);
					if (fabs(euler.x) < 1e-3f)
					{
						euler.x = 0;
					}
					if (fabs(euler.y) < 1e-3f)
					{
						euler.y = 0;
					}
					if (fabs(euler.z) < 1e-3f)
					{
						euler.z = 0;
					}
					transform->SetLocalEulerRotationHint(euler);
				}
			}
		}
		break;
		case 2:
		{
			if (Gizmos::ScaleHandle(position, rotation, scale, isGizmoLocal))
			{
				Vector3 delta = scale - transform->GetLocalScale();
				for (Object* target : m_SerializedObject->GetTargets())
				{
					Transform* targetTransform = static_cast<Transform*>(target);
					UndoManager::RecordObject(targetTransform);
					targetTransform->SetLocalScale(targetTransform->GetLocalScale() + delta);
				}
			}
		}
		break;
		}
	}

	//Transform* parentTransform = transform->GetParent();

	//Matrix transformMatrix = Matrix::Identity;
	//if (parentTransform != nullptr)
	//{
	//	transformMatrix = parentTransform->GetLocalToWorldMatrix();
	//}

	//Gizmos::SetMatrix(transformMatrix);

	//void TransformEditor::OnDrawSceneSelected()
	//{
	//	// TODO update with SetLocalPosition() and targets instead
	//	
	//	//Matrix transformMatrix = Matrix::Identity;
	//	
	//	/*Matrix transformMatrix = Matrix::CreateScale(transform->GetLocalScale()) * Matrix::CreateFromQuaternion(transform->GetLocalRotation()) * Matrix::CreateTranslation(transform->GetLocalPosition());
	//	if (parentTransform != nullptr)
	//	{
	//		transformMatrix *= parentTransform->GetLocalToWorldMatrix();
	//	}*/

	//	
	//	

	//	Camera* camera = Camera::GetCurrent();
	//	if (camera != nullptr)
	//	{
	//		float snapping[3];
	//		float* gizmoSnapping = Preferences::GetGizmoSnapping();
	//		int gizmoOperation = Preferences::GetGizmoOperation();
	//		if (gizmoOperation == ImGuizmo::OPERATION::TRANSLATE)
	//		{
	//			snapping[0] = gizmoSnapping[0];
	//			snapping[1] = gizmoSnapping[0];
	//			snapping[2] = gizmoSnapping[0];
	//		}
	//		else if (gizmoOperation == ImGuizmo::OPERATION::ROTATE)
	//		{
	//			snapping[0] = gizmoSnapping[1];
	//			snapping[1] = gizmoSnapping[1];
	//			snapping[2] = gizmoSnapping[1];
	//		}
	//		else if (gizmoOperation == ImGuizmo::OPERATION::SCALE)
	//		{
	//			snapping[0] = gizmoSnapping[2];
	//			snapping[1] = gizmoSnapping[2];
	//			snapping[2] = gizmoSnapping[2];
	//		}
	//		
	//		/*if (ImGuizmo::Manipulate((float*)camera->GetViewMatrix().m, (float*)camera->GetProjectionMatrix().m, (ImGuizmo::OPERATION)gizmoOperation, ImGuizmo::MODE::WORLD, (float*)transformMatrix.m, 0, snapping))
	//		{
	//			UndoManager::RecordObject(transform);

	//			Vector3 scale;
	//			Quaternion rotation;
	//			Vector3 translation;

	//			if (parentTransform != nullptr)
	//			{
	//				transformMatrix *= parentTransform->GetLocalToWorldMatrix().Invert();
	//			}
	//			
	//			transformMatrix.Decompose(scale, rotation, translation);
	//			if (gizmoOperation == ImGuizmo::OPERATION::TRANSLATE && !Math::Approximately(Math::RoundToN(translation, 4), Math::RoundToN(m_LocalPositionProperty.GetVector3(), 4)))
	//			{
	//				transform->SetLocalPosition(translation);
	//			}
	//			if (gizmoOperation == ImGuizmo::OPERATION::ROTATE && !Math::Approximately(Math::RoundToN(rotation, 4), Math::RoundToN(m_LocalRotationProperty.GetQuaternion(), 4)))
	//			{
	//				transform->SetLocalRotation(rotation, false);

	//				Vector3 previousEuler = transform->GetLocalEulerRotationHint();
	//				Vector3 euler = CorrectEuler(previousEuler, Math::ToDegrees(rotation.ToEuler()));

	//				if (snapping[1] > 0)
	//				{
	//					euler /= snapping[1];
	//					euler = Vector3(roundf(euler.x), roundf(euler.y), roundf(euler.z));
	//					euler *= snapping[1];
	//				}

	//				if (fabs(euler.x) < 1e-6f)
	//				{
	//					euler.x = 0;
	//				}
	//				if (fabs(euler.y) < 1e-6f)
	//				{
	//					euler.y = 0;
	//				}
	//				if (fabs(euler.z) < 1e-6f)
	//				{
	//					euler.z = 0;
	//				}
	//				transform->SetLocalEulerRotationHint(euler);
	//			}
	//			if (gizmoOperation == ImGuizmo::OPERATION::SCALE && !Math::Approximately(Math::RoundToN(scale, 4), Math::RoundToN(m_LocalScaleProperty.GetVector3(), 4)))
	//			{
	//				transform->SetLocalScale(scale);
	//			}
	//		}*/
	//	}
	//}

	void TransformEditor::OnHierarchyUpdate()
	{
		if (m_SerializedObject->IsValid())
		{
			m_SerializedObject->Update();
		}
	}

	void TransformEditor::OnPropertyClear(ImGui::ClearOverrideEventArgs& args)
	{
		SerializedProperty* property = args.GetProperty();
		SerializedObject* serializedObject = property->GetSerializedObject();
		if (serializedObject->GetTarget()->IsClassType(Transform::Type) && property->GetName() == "m_LocalRotationEulerHint")
		{
			serializedObject->FindProperty("m_LocalRotation").ClearOverride();
		}
	}
}