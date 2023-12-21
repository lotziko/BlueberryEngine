#pragma once

namespace ImGui
{
	bool DragVector3(const char* label, Blueberry::Vector3& vector);
	bool ColorEdit(const char* label, Blueberry::Color& color);
	template<class ObjectType>
	bool ObjectEdit(const char* label, ObjectType* object);

	void ApplyEditorDarkTheme();
	void LoadDefaultEditorFonts();

	template<class ObjectType>
	bool ObjectEdit(const char* label, ObjectType* object)
	{
		ImGui::PushID(label);

		ImGui::Text(label);
		ImGui::SameLine();
		ImGui::SetCursorPosX(100);

		if (ImGui::BeginDragDropTarget())
		{
			if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ASSET_OBJECT"))
			{
				BB_INFO("Drop");
			}
			ImGui::EndDragDropTarget();
		}

		ImGui::PopID();
		return false;
	}
}