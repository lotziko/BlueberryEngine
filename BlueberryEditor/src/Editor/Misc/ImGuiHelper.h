#pragma once
#include "Blueberry\Core\ObjectDB.h"

namespace ImGui
{
	bool DragVector3(const char* label, Blueberry::Vector3& vector);
	bool ColorEdit(const char* label, Blueberry::Color& color);
	template<class ObjectType>
	bool ObjectEdit(const char* label, ObjectType*& object);

	void ApplyEditorDarkTheme();
	void LoadDefaultEditorFonts();

	template<class ObjectType>
	bool ObjectEdit(const char* label, ObjectType*& object)
	{
		static_assert(std::is_base_of<Blueberry::Object, ObjectType>::value, "Type is not derived from Object.");
		bool result = false;

		ImGui::PushID(label);

		ImGui::Text(label);
		ImGui::SameLine();
		ImGui::SetCursorPosX(100);

		if (ImGui::BeginDragDropTarget())
		{
			if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("OBJECT_ID"))
			{
				Blueberry::ObjectId* id = (Blueberry::ObjectId*)payload->Data;
				Blueberry::ObjectItem* item = Blueberry::ObjectDB::IdToObjectItem(*id);
				if (item != nullptr)
				{
					object = (ObjectType*)item->object;
					result = true;
				}
			}
			ImGui::EndDragDropTarget();
		}
		ImGui::Text(object != nullptr ? object->GetName().c_str() : "None");

		ImGui::PopID();
		return result;
	}
}