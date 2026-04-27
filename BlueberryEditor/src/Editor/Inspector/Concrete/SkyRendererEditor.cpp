#include "SkyRendererEditor.h"

#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Scene\Components\SkyRenderer.h"
#include "Editor\Assets\Processors\ReflectionGenerator.h"
#include "Editor\Panels\Scene\SceneArea.h"
#include "Editor\Misc\ImGuiHelper.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	void SkyRendererEditor::OnEnable()
	{
		m_MaterialProperty = m_SerializedObject->FindProperty("m_Material");
	}

	void SkyRendererEditor::OnDrawInspector()
	{
		ImGui::EditorStyle& style = ImGui::GetEditorStyle();

		ObjectEditor::OnDrawInspector();
		if (m_SerializedObject->GetTargets().size() == 1 && ImGui::Button("Bake"))
		{
			if (m_MaterialProperty.GetObjectPtr().Get() != nullptr)
			{
				SkyRenderer* skyRenderer = static_cast<SkyRenderer*>(m_SerializedObject->GetTarget());
				ReflectionGenerator::GenerateReflectionTexture(skyRenderer);
				SceneArea::RequestRedrawAll();
			}
		}

		m_SerializedObject->ApplyModifiedProperties();
	}
}
