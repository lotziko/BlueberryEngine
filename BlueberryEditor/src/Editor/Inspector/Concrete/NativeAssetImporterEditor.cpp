#include "NativeAssetImporterEditor.h"

#include "Editor\Assets\AssetDB.h"
#include "Editor\Misc\ImGuiHelper.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	void NativeAssetImporterEditor::OnDrawInspector()
	{
		ImGui::BeginChangeCheck();
		AssetImporterEditor::OnDrawInspector();
		if (ImGui::EndChangeCheck())
		{
			for (Object* object : m_SerializedObject->GetTargets())
			{
				AssetDB::SetDirty(object);
			}
		}
	}
}
