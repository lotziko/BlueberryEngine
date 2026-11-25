#include "ReflectionProbeEditor.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\ReflectionProbe.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Editor\Graphics\ReflectionGenerator.h"
#include "Editor\Panels\Scene\SceneArea.h"
#include "Editor\EditorSceneManager.h"
#include "Editor\Gizmos\Gizmos.h"
#include "Editor\Misc\ImGuiHelper.h"
#include "Editor\Undo\UndoManager.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	static Texture* s_Icon = nullptr;
	static bool s_EditBounds = false;

	ReflectionProbeEditor::ReflectionProbeEditor()
	{
		if (s_Icon == nullptr)
		{
			s_Icon = static_cast<Texture*>(AssetLoader::Load("assets/icons/ReflectionProbeIcon.png"));
		}
	}

	void ReflectionProbeEditor::OnEnable()
	{
		m_TypeProperty = m_SerializedObject->FindProperty("m_Type");
		m_RadiusProperty = m_SerializedObject->FindProperty("m_Radius");
		m_SizeProperty = m_SerializedObject->FindProperty("m_Size");
		m_FadeProperty = m_SerializedObject->FindProperty("m_Fade");
	}

	void ReflectionProbeEditor::OnDisable()
	{
		s_EditBounds = false;
	}

	void ReflectionProbeEditor::OnDrawInspector()
	{
		m_SerializedObject->UpdateIfNeeded();

		if (m_SerializedObject->GetTargetCount() == 1)
		{
			if (s_EditBounds)
			{
				if (ImGui::Button("Stop edit bounds"))
				{
					s_EditBounds = false;
				}
			}
			else
			{
				if (ImGui::Button("Start edit bounds"))
				{
					s_EditBounds = true;
				}
			}
		}

		ImGui::Property(&m_TypeProperty, "Type");

		ReflectionProbeType type = m_TypeProperty.GetEnum<ReflectionProbeType>();
		if (type == ReflectionProbeType::Sphere)
		{
			ImGui::Property(&m_RadiusProperty, "Radius");
		}
		else
		{
			ImGui::Property(&m_SizeProperty, "Size");
		}

		ImGui::Property(&m_FadeProperty, "Fade");

		if (ImGui::Button("Bake"))
		{
			for (Object* target : m_SerializedObject->GetTargets())
			{
				ReflectionProbe* reflectionProbe = static_cast<ReflectionProbe*>(target);
				ReflectionGenerator::GenerateReflectionTexture(reflectionProbe);
			}
			SceneArea::RequestRedrawAll();
		}

		m_SerializedObject->ApplyModifiedProperties();
	}

	Texture* ReflectionProbeEditor::GetIcon(Object* object)
	{
		return s_Icon;
	}

	void ReflectionProbeEditor::OnDrawSceneSelected()
	{
		for (Object* target : m_SerializedObject->GetTargets())
		{
			ReflectionProbe* reflectionProbe = static_cast<ReflectionProbe*>(target);
			Transform* transform = reflectionProbe->GetTransform();
			Vector3 position = Vector3::Zero;

			Gizmos::SetMatrix(transform->GetLocalToWorldMatrix());
			if (reflectionProbe->GetType() == ReflectionProbeType::Sphere)
			{
				float radius = reflectionProbe->GetRadius();
				Gizmos::DrawSphere(position, radius);
				if (s_EditBounds)
				{
					if (Gizmos::SphereBoundsHandle(position, radius))
					{
						UndoManager::RecordObject(reflectionProbe);
						UndoManager::RecordObject(transform);
						transform->SetPosition(transform->GetPosition() + Math::MultiplyVector(transform->GetLocalToWorldMatrix(), position));
						reflectionProbe->SetRadius(radius);
					}
				}
			}
			else
			{
				Vector3 size = reflectionProbe->GetSize();
				Gizmos::DrawBox(position, size);
				if (s_EditBounds)
				{
					if (Gizmos::BoxBoundsHandle(position, size))
					{
						UndoManager::RecordObject(reflectionProbe);
						UndoManager::RecordObject(transform);
						transform->SetPosition(transform->GetPosition() + Math::MultiplyVector(transform->GetLocalToWorldMatrix(), position));
						reflectionProbe->SetSize(size);
					}
				}
			}
		}
	}
}
