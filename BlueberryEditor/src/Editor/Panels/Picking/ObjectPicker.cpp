#include "ObjectPicker.h"

#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Core\Screen.h"
#include "Blueberry\Graphics\Texture2D.h"

#include "Editor\Selection.h"
#include "Editor\Assets\IconDB.h"
#include "Editor\Assets\ThumbnailCache.h"
#include "Editor\Misc\ImGuiHelper.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	static void* s_IdPtr = nullptr;
	static const char* s_Label;
	static std::string s_SearchName;
	static List<Object*> s_AllObjects = {};
	static List<Object*> s_Objects = {};
	static Object* s_SelectedObject = nullptr;

	void ObjectPicker::Open(Object** object, const size_t& type)
	{
		s_IdPtr = object;
		s_Label = Blueberry::ClassDB::GetInfo(type).name.c_str();
		s_SearchName.clear();
		s_AllObjects.clear();
		s_Objects.clear();
		s_AllObjects.emplace_back(nullptr);
		ObjectDB::GetObjects(type, s_AllObjects, true);
		Object* objectValue = *object;
		for (auto it = s_AllObjects.begin(); it < s_AllObjects.end(); ++it)
		{
			s_Objects.emplace_back(*it);
		}
		for (auto it = s_AllObjects.begin(); it < s_AllObjects.end(); ++it)
		{
			if (*it == objectValue)
			{
				s_SelectedObject = objectValue;
				break;
			}
		}
		ImGui::PushID(object);
		ImGui::OpenPopup(s_Label);
		ImGui::PopID();
	}

	bool ObjectPicker::GetResult(Object** object)
	{
		ImGui::EditorStyle& style = ImGui::GetEditorStyle();

		if (object != s_IdPtr)
		{
			return false;
		}
		bool open = true;
		bool hasResult = false;
		ImGui::PushStyleColor(ImGuiCol_PopupBg, ImGui::GetColorU32(ImGuiCol_WindowBg));
		ImGui::PushStyleColor(ImGuiCol_ModalWindowDimBg, ImVec4(0, 0, 0, 0));
		ImGui::PushID(object);
		if (ImGui::BeginPopupModal(s_Label, &open))
		{
			ImGui::SetWindowSize(ImVec2(700 * Screen::GetScale(), 500 * Screen::GetScale()), ImGuiCond_FirstUseEver);
			
			ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPosX() + 2, ImGui::GetCursorPosY() + 1));
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 2);
			if (ImGui::SearchInputText("Search..", &s_SearchName))
			{
				String name = String(s_SearchName);
				s_Objects.clear();
				s_Objects.emplace_back(s_AllObjects[0]);
				for (auto it = s_AllObjects.begin() + 1; it < s_AllObjects.end(); ++it)
				{
					if (StringHelper::HasSubstring((*it)->GetName(), name))
					{
						s_Objects.emplace_back(*it);
					}
				}
			}
			ImGui::PopItemWidth();

			ImGui::BeginChild("Content");
			ImVec2 pos = ImGui::GetCursorPos();
			ImVec2 size = ImGui::GetContentRegionAvail();

			uint32_t maxCells = static_cast<uint32_t>(floorf(size.x / (style.ProjectCellSize + style.ProjectSpaceBetweenCells)));
			if (maxCells > 0)
			{
				float expandedSpaceBetweenCells = (size.x - (maxCells * style.ProjectCellSize)) / (maxCells + 1);
				uint32_t cellIndex = 0;

				for (uint32_t i = 0; i < s_Objects.size(); ++i)
				{
					pos.x += expandedSpaceBetweenCells;

					ImGui::SetCursorPos(pos);
					Object* currentObject = s_Objects[i];
					bool selected = s_SelectedObject == currentObject;
					if (i > 0)
					{
						if (DrawObject(currentObject, selected))
						{
							*object = currentObject;
							s_SelectedObject = currentObject;
							hasResult = true;
						}
					}
					else
					{
						if (DrawNone(selected))
						{
							*object = nullptr;
							s_SelectedObject = nullptr;
							hasResult = true;
						}
					}
					if (selected && ImGui::IsItemClicked(0))
					{
						open = false;
					}

					if (cellIndex + 1 < maxCells)
					{
						pos.x += style.ProjectCellSize;
						++cellIndex;
					}
					else
					{
						pos.y += style.ProjectCellSize + style.ProjectSpaceBetweenCells;
						pos.x = 0;
						cellIndex = 0;
					}
				}
			}
			if (!open)
			{
				ImGui::CloseCurrentPopup();
				s_IdPtr = nullptr;
				s_Label = nullptr;
				s_SelectedObject = nullptr;
			}
			ImGui::EndChild();
			ImGui::EndPopup();
		}
		ImGui::PopID();
		ImGui::PopStyleColor(2);
		return hasResult;
	}

	bool ObjectPicker::DrawNone(const bool& selected)
	{
		ImGui::EditorStyle& style = ImGui::GetEditorStyle();

		bool isSelected = false;
		ImVec2 screenPos = ImGui::GetCursorScreenPos();

		ImGui::PushID(ImGuiID("None"));
		ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, ImVec2(0.5f, 1.0f));

		if (ImGui::Selectable("None", selected, ImGuiSelectableFlags_DontClosePopups, ImVec2(style.ProjectCellSize, style.ProjectCellSize)))
		{
			isSelected = true;
		}

		ImGui::PopStyleVar();
		ImGui::PopID();
		return isSelected;
	}

	bool ObjectPicker::DrawObject(Object* object, const bool& selected)
	{
		ImGui::EditorStyle& style = ImGui::GetEditorStyle();

		bool isSelected = false;
		ImVec2 screenPos = ImGui::GetCursorScreenPos();

		ImGui::PushID(object->GetObjectId());
		ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, ImVec2(0.5f, 1.0f));

		if (ImGui::Selectable(object->GetName().c_str(), selected, ImGuiSelectableFlags_DontClosePopups, ImVec2(style.ProjectCellSize, style.ProjectCellSize)))
		{
			isSelected = true;
		}

		Texture* icon = ThumbnailCache::GetThumbnail(object);
		if (icon == nullptr)
		{
			icon = IconDB::GetAssetIcon(object);
		}
		ImGui::GetWindowDrawList()->AddImage(reinterpret_cast<ImTextureID>(icon->GetHandle()), ImVec2(screenPos.x + style.ProjectCellIconPadding, screenPos.y), ImVec2(screenPos.x + style.ProjectCellSize - style.ProjectCellIconPadding, screenPos.y + style.ProjectCellSize - style.ProjectCellIconPadding * 2), ImVec2(0, 1), ImVec2(1, 0));

		ImGui::PopStyleVar();
		ImGui::PopID();
		return isSelected;
	}
}
