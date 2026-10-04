#include "ComponentPicker.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Scene\Components\Transform.h"

#include "Editor\Misc\ImGuiHelper.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	bool ShouldHideType(const ClassInfo& info)
	{
		if (info.createInstance == nullptr)
		{
			return true;
		}

		if (info.id == Transform::Type)
		{
			return true;
		}

		if (!ClassDB::IsParent(info.id, Component::Type))
		{
			return true;
		}

		return false;
	}

#define INVALID_TYPE UINT32_MAX
#define INVALID_INDEX UINT32_MAX

	struct Entry
	{
		String name;
		String displayName;
		TypeId type = INVALID_TYPE;
		uint32_t parent = INVALID_INDEX;
		bool isCategory;
		List<uint32_t> children;
	};

	static String s_SearchName;
	static List<int> s_SearchIndexes = {};
	static uint32_t s_PreviousParent = 0;
	static uint32_t s_CurrentParent = 0;
	static List<Entry> s_Entries = {};
	static float s_AnimationProgress = 1.0f;
	static float s_AnimationDirection = 0.0f;

	uint32_t AddEntry(const String& name, TypeId type, uint32_t parentIndex, bool isCategory)
	{
		Entry entry = {};
		entry.name = name;
		entry.displayName = StringHelper::ToDisplayName(name);
		entry.type = type;
		entry.parent = parentIndex;
		entry.isCategory = isCategory;
		uint32_t index = static_cast<uint32_t>(s_Entries.size());
		s_Entries.push_back(std::move(entry));
		if (parentIndex != INVALID_INDEX)
		{
			s_Entries[parentIndex].children.push_back(index);
		}
		return index;
	}

	void ComponentPicker::Open()
	{
		ImGui::OpenPopup("componentPicker");

		s_SearchName.clear();
		s_SearchIndexes.clear();
		s_PreviousParent = 0;
		s_CurrentParent = 0;
		s_Entries.clear();
		s_AnimationProgress = 1.0f;
		s_AnimationDirection = 0.0f;

		uint32_t root = AddEntry("Component", INVALID_TYPE, INVALID_INDEX, true);
		auto& infos = ClassDB::GetInfos();
		for (auto& info : infos)
		{
			if (ShouldHideType(info))
			{
				continue;
			}

			String fullPath = info.addComponentMenuPath.empty() ? (info.isDll ? String("Scripts/").append(info.name) : info.name) : info.addComponentMenuPath;

			uint32_t parent = 0;
			size_t start = 0;
			size_t separator;
			while ((separator = fullPath.find('/', start)) != String::npos)
			{
				const String categoryName = fullPath.substr(start, separator - start);
				uint32_t category = INVALID_INDEX;
				for (int child : s_Entries[parent].children)
				{
					const Entry& entry = s_Entries[child];
					if (entry.isCategory && entry.name == categoryName)
					{
						category = child;
						break;
					}
				}
				if (category == INVALID_INDEX)
				{
					category = AddEntry(categoryName.c_str(), 0, parent, true);
				}
				parent = category;
				start = separator + 1;
			}
			AddEntry(fullPath.substr(start), info.id, parent, false);
		}
	}

	uint32_t DrawPage(uint32_t parent, const ImVec2& size)
	{
		uint32_t clickedIndex = INVALID_INDEX;
		ImGuiWindowFlags flags = ImGuiWindowFlags_NoBackground;
		ImGui::PushID(parent);
		ImGui::BeginGroup();

		ImVec2 pos = ImGui::GetCursorScreenPos();
		float padding = 5.0f;
		ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0.0f);
		if (ImGui::Button(s_Entries[parent].name.c_str(), ImVec2(size.x, 0)))
		{
			if (parent != 0)
			{
				clickedIndex = s_Entries[parent].parent;
			}
		}
		if (parent != 0)
		{
			ImGui::GetWindowDrawList()->AddText(ImVec2(pos.x + padding - 2, pos.y), ImGui::GetColorU32(ImGuiCol_Text), "<");
		}
		ImGui::PopStyleVar();

		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(padding, 0.0f));
		if (ImGui::BeginChild("page", size, ImGuiChildFlags_AlwaysUseWindowPadding, flags))
		{
			const Entry& parentEntry = s_Entries[parent];
			for (uint32_t index : parentEntry.children)
			{
				pos = ImGui::GetCursorScreenPos();
				ImVec2 size = ImGui::GetContentRegionAvail();
				const Entry& entry = s_Entries[index];
				if (ImGui::Selectable(entry.displayName.c_str(), false, ImGuiSelectableFlags_NoAutoClosePopups))
				{
					clickedIndex = index;
				}
				if (entry.isCategory)
				{
					ImGui::GetWindowDrawList()->AddText(ImVec2(pos.x + size.x - ImGui::CalcTextSize(">").x, pos.y), ImGui::GetColorU32(ImGuiCol_Text), ">");
				}
			}
			ImGui::EndChild();
			ImGui::PopStyleVar();
		}

		ImGui::EndGroup();
		ImGui::PopID();
		return clickedIndex;
	}

	uint32_t DrawSearchPage(const ImVec2& size)
	{
		uint32_t clickedIndex = INVALID_INDEX;
		ImGuiWindowFlags flags = ImGuiWindowFlags_NoBackground;
		if (ImGui::BeginChild("search page", size, ImGuiChildFlags_None, flags))
		{
			for (uint32_t index : s_SearchIndexes)
			{
				const Entry& entry = s_Entries[index];
				if (ImGui::Selectable(entry.displayName.c_str(), false, ImGuiSelectableFlags_NoAutoClosePopups))
				{
					clickedIndex = index;
				}
			}
			ImGui::EndChild();
		}
		return clickedIndex;
	}

	void NavigateTo(uint32_t parentIndex, float direction)
	{
		s_PreviousParent = s_CurrentParent;
		s_CurrentParent = parentIndex;
		s_AnimationProgress = 0.0f;
		s_AnimationDirection = direction;
	}

	bool ComponentPicker::GetResult(TypeId* type)
	{
		bool result = false;
		ImGui::SetNextWindowSize(ImVec2(320, 390));
		if (ImGui::BeginPopup("componentPicker", ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse))
		{
			ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPosX() + 2, ImGui::GetCursorPosY() + 2));
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 2);
			if (ImGui::SearchInputText("Search..", &s_SearchName))
			{
				String name = String(s_SearchName);
				s_SearchIndexes.clear();
				if (!name.empty())
				{
					for (size_t i = 0; i < s_Entries.size(); ++i)
					{
						const Entry& entry = s_Entries[i];
						if (!entry.isCategory && StringHelper::HasSubstring(entry.name, name))
						{
							s_SearchIndexes.push_back(i);
						}
					}
				}
			}

			ImVec2 pos = ImGui::GetCursorPos();
			ImVec2 size = ImGui::GetContentRegionAvail();
			uint32_t clickedIndex = INVALID_INDEX;
			if (s_SearchName.empty())
			{
				s_AnimationProgress = std::min(1.0f, s_AnimationProgress + ImGui::GetIO().DeltaTime / 0.18f);
				uint32_t previousParent = s_PreviousParent;
				uint32_t parent = s_CurrentParent;
				bool animating = s_AnimationProgress < 1.0f;
				if (animating)
				{
					float t = s_AnimationProgress;
					float eased = t * t * (3.0f - 2.0f * t);
					ImGui::SetCursorPos(ImVec2(pos.x - s_AnimationDirection * eased * size.x, pos.y));
					DrawPage(previousParent, size);
					ImGui::SetCursorPos(ImVec2(pos.x + s_AnimationDirection * (1.0f - eased) * size.x, pos.y));
					DrawPage(parent, size);
				}
				else
				{
					uint32_t index = DrawPage(parent, size);
					if (index != INVALID_INDEX)
					{
						clickedIndex = index;
					}
				}
				if (clickedIndex != INVALID_INDEX && s_Entries[clickedIndex].isCategory)
				{
					NavigateTo(clickedIndex, clickedIndex == s_Entries[parent].parent ? -1.0f : 1.0f);
				}
			}
			else
			{
				clickedIndex = DrawSearchPage(size);
			}

			if (clickedIndex != INVALID_INDEX)
			{
				const Entry& entry = s_Entries[clickedIndex];
				if (!entry.isCategory)
				{
					*type = entry.type;
					result = true;
					ImGui::CloseCurrentPopup();
				}
			}

			ImGui::EndPopup();
		}
		return result;
	}
}