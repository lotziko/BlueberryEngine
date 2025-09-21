#include "EntityEditor.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Component.h"

#include "imgui\imgui.h"
#include "Blueberry\Core\ClassDB.h"

#include "Editor\Inspector\ObjectEditorDB.h"
#include "Editor\Panels\Inspector\InspectorExpandedItemsCache.h"
#include "Editor\Panels\Scene\SceneArea.h"
#include "Editor\Misc\ImGuiHelper.h"
#include "Editor\EditorObjectManager.h"
#include "Editor\Selection.h"

namespace Blueberry
{
	bool ShouldHideType(const size_t type, const ClassInfo& info)
	{
		if (info.createInstance == nullptr)
		{
			return true;
		}

		if (!ClassDB::IsParent(type, Component::Type))
		{
			return true;
		}

		return false;
	}

	void EntityEditor::OnEnable()
	{
		m_HasPadding = false;
		m_IsActiveProperty = m_SerializedObject->FindProperty("m_IsActive");
		m_ComponentsProperty = m_SerializedObject->FindProperty("m_Components");
		
		auto& targets = m_SerializedObject->GetTargets();
		Dictionary<size_t, List<Object*>> components;
		for (Object* target : targets)
		{
			Entity* entity = static_cast<Entity*>(target);
			for (uint32_t i = 0; i < entity->GetComponentCount(); ++i)
			{
				Component* component = entity->GetComponent(i);
				size_t type = component->GetType();
				components[type].emplace_back(component);
			}
		}
		
		for (auto& pair : components)
		{
			auto& list = pair.second;
			if (list.size() == targets.size())
			{
				ObjectEditor* editor = ObjectEditor::GetEditor(list);
				if (editor != nullptr)
				{
					m_ComponentsEditors.emplace_back(std::make_pair(list[0], editor));
				}
			}
		}

		EditorObjectManager::GetEntityDestroyed().AddCallback<EntityEditor, &EntityEditor::OnEntityDestroy>(this);
	}

	void EntityEditor::OnDisable()
	{
		if (m_ComponentsEditors.size() > 0)
		{
			for (auto& pair : m_ComponentsEditors)
			{
				ObjectEditor::ReleaseEditor(pair.second);
			}
			m_ComponentsEditors.clear();
		}

		EditorObjectManager::GetEntityDestroyed().RemoveCallback<EntityEditor, &EntityEditor::OnEntityDestroy>(this);
	}

	void EntityEditor::OnDrawInspector()
	{
		ImGui::Property(&m_IsActiveProperty, "Is Active");

		for (auto& pair : m_ComponentsEditors)
		{
			String name = pair.first->GetTypeName();
			const char* headerId = name.c_str();
			const char* popupId = "InspectorPopup";

			ImGui::PushID(headerId);

			ImGui::SetNextItemOpen(InspectorExpandedItemsCache::Get(name));
			bool opened = ImGui::CollapsingHeader(headerId);
			if (ImGui::IsItemToggledOpen())
			{
				InspectorExpandedItemsCache::Set(name, opened);
			}

			if (opened)
			{
				pair.second->DrawInspector();
			}

			if (ImGui::BeginPopup(popupId))
			{
				if (ImGui::MenuItem("Remove component"))
				{
				}
				ImGui::EndPopup();
			}

			ImGui::OpenPopupOnItemClick(popupId, ImGuiPopupFlags_MouseButtonRight);

			ImGui::PopID();
		}

		if (m_SerializedObject->ApplyModifiedProperties())
		{
			SceneArea::RequestRedrawAll();
		}

		if (ImGui::Button("Add component..."))
		{
			ImGui::OpenPopup("addComponent");
		}

		if (ImGui::BeginPopup("addComponent"))
		{
			auto infos = ClassDB::GetInfos();

			for (auto info : infos)
			{
				if (ShouldHideType(info.first, info.second))
				{
					continue;
				}

				if (ImGui::Selectable(info.second.name.c_str()))
				{
					for (Object* target : m_SerializedObject->GetTargets())
					{
						(static_cast<Entity*>(target))->AddComponent(static_cast<Component*>(info.second.createInstance()));
					}
					OnDisable();
					OnEnable();
					SceneArea::RequestRedrawAll();
				}
			}

			ImGui::EndPopup();
		}
	}

	void EntityEditor::OnEntityDestroy()
	{
		Selection::SetActiveObject(nullptr);
	}
}