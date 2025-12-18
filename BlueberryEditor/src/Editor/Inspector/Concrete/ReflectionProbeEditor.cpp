#include "ReflectionProbeEditor.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\ReflectionProbe.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Editor\Assets\Processors\ReflectionGenerator.h"
#include "Editor\Panels\Scene\SceneArea.h"
#include "Editor\EditorSceneManager.h"
#include "Editor\Gizmos\Gizmos.h"
#include "Editor\Misc\ImGuiHelper.h"

#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\Texture2D.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	static Texture* s_Icon = nullptr;

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

	void ReflectionProbeEditor::OnDrawInspector()
	{
		if (m_SerializedObject->GetTargets().size() == 1)
		{
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

			ImGui::EditorStyle& style = ImGui::GetEditorStyle();
			ImGui::Indent(style.InspectorIndent);
			if (ImGui::Button("Bake"))
			{
				Scene* scene = EditorSceneManager::GetScene();
				if (scene != nullptr)
				{
					ReflectionProbe* reflectionProbe = static_cast<ReflectionProbe*>(m_SerializedObject->GetTarget());
					ReflectionGenerator::GenerateReflectionTexture(reflectionProbe);
					SceneArea::RequestRedrawAll();
				}
			}
			ImGui::Unindent(style.InspectorIndent);
		}

		if (m_SerializedObject->ApplyModifiedProperties())
		{
			SceneArea::RequestRedrawAll();
		}
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

			Gizmos::SetMatrix(Matrix::Identity);
			if (reflectionProbe->GetType() == ReflectionProbeType::Sphere)
			{
				Gizmos::DrawSphere(transform->GetPosition(), reflectionProbe->GetRadius());
			}
			else
			{
				Gizmos::DrawBox(transform->GetPosition(), reflectionProbe->GetSize());
			}
		}
	}
}
