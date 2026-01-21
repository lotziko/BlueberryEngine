#include "AnimationGraphEditor.h"

#include "Editor\Assets\AssetDB.h"
#include "Editor\Misc\ImGuiHelper.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	void AnimationGraphEditor::OnDrawInspector()
	{
		ImGui::BeginPaddedArea(ImVec2(10, 5), ImVec2(10, 5));
		ObjectEditor::OnDrawInspector();
		if (ImGui::Button("Save"))
		{
			for (Object* object : m_SerializedObject->GetTargets())
			{
				AssetDB::SetDirty(object);
			}
			AssetDB::SaveAssets();
			AssetDB::Refresh();
		}
		ImGui::EndPaddedArea();
	}
}
