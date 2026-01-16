#include "AnimationGraphEditor.h"

#include "Editor\Assets\AssetDB.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	void AnimationGraphEditor::OnDrawInspector()
	{
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
	}
}
