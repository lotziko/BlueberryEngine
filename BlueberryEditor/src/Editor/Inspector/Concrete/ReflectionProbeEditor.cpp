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
		m_SizeProperty = m_SerializedObject->FindProperty("m_Size");
	}

	void ReflectionProbeEditor::OnDrawInspector()
	{
		ImGui::Property(&m_SizeProperty, "Size");

		if (m_SerializedObject->GetTargets().size() == 1 && ImGui::Button("Bake"))
		{
			Scene* scene = EditorSceneManager::GetScene();
			if (scene != nullptr)
			{
				ReflectionProbe* reflectionProbe = static_cast<ReflectionProbe*>(m_SerializedObject->GetTarget());
				ReflectionGenerator::GenerateReflectionTexture(reflectionProbe);
				SceneArea::RequestRedrawAll();
			}
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
			Gizmos::DrawBox(transform->GetPosition(), reflectionProbe->GetSize());
		}
	}
}
