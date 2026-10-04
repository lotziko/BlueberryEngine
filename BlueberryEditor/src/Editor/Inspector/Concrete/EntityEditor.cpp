#include "EntityEditor.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Component.h"

#include "Blueberry\Core\ClassDB.h"

#include "Editor\Panels\Inspector\InspectorExpandedItemsCache.h"
#include "Editor\Panels\Picking\ComponentPicker.h"
#include "Editor\Misc\ImGuiHelper.h"
#include "Editor\EditorObjectManager.h"
#include "Editor\Selection.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	bool EntityEditor::IsInspectorPadded()
	{
		return false;
	}

	void EntityEditor::OnEnable()
	{
		m_IsActiveProperty = m_SerializedObject->FindProperty("m_IsActive");
		m_ComponentsProperty = m_SerializedObject->FindProperty("m_Components");
		
		auto& targets = m_SerializedObject->GetTargets();
		List<std::pair<size_t, List<Object*>>> components;
		for (Object* target : targets)
		{
			Entity* entity = static_cast<Entity*>(target);
			for (size_t i = 0; i < entity->GetComponentCount(); ++i)
			{
				Component* component = entity->GetComponentAt(i);
				TypeId type = component->GetType();
				bool foundPlace = false;
				for (size_t j = 0; j < components.size(); ++j)
				{
					auto& pair = components[i];
					if (pair.first == type)
					{
						pair.second.push_back(component);
						foundPlace = true;
						break;
					}
				}
				if (!foundPlace)
				{
					List<Object*> list;
					list.push_back(component);
					components.push_back(std::make_pair(type, std::move(list)));
				}
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
					Object* component = list[0];
					m_ComponentsEditors.push_back({ component->GetObjectId(), component->GetTypeName(), StringHelper::ToDisplayName(component->GetTypeName()), editor });
				}
			}
		}

		EditorObjectManager::GetEntityDestroyed().AddCallback<EntityEditor, &EntityEditor::OnEntityDestroy>(this);
	}

	void EntityEditor::OnDisable()
	{
		if (m_ComponentsEditors.size() > 0)
		{
			for (auto& componentEditor : m_ComponentsEditors)
			{
				ObjectEditor::ReleaseEditor(componentEditor.editor);
			}
			m_ComponentsEditors.clear();
		}

		EditorObjectManager::GetEntityDestroyed().RemoveCallback<EntityEditor, &EntityEditor::OnEntityDestroy>(this);
	}

	void EntityEditor::OnDrawInspector()
	{
		ImGui::BeginPaddedArea(ImVec2(18, 5), ImVec2(10, 5));
		ImGui::Property(&m_IsActiveProperty, "Is Active");
		ImGui::EndPaddedArea();

		for (auto& componentEditor : m_ComponentsEditors)
		{
			ImGui::PushID(componentEditor.id);

			ImGui::SetNextItemOpen(InspectorExpandedItemsCache::Get(componentEditor.name));
			ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0.0f);
			bool opened = ImGui::CollapsingHeader(componentEditor.displayName.c_str());
			ImGui::PopStyleVar();
			if (ImGui::IsItemToggledOpen())
			{
				InspectorExpandedItemsCache::Set(componentEditor.name, opened);
			}

			if (ImGui::BeginPopupContextItem())
			{
				if (ImGui::MenuItem("Remove component"))
				{
					for (Object* target : componentEditor.editor->GetSerializedObject()->GetTargets())
					{
						m_RemovedComponents.push_back(static_cast<Component*>(target));
					}
				}
				ImGui::EndPopup();
			}

			if (opened)
			{
				ImGui::BeginPaddedArea(ImVec2(18, 5), ImVec2(10, 5));
				componentEditor.editor->DrawInspector();
				ImGui::EndPaddedArea();
			}

			ImGui::PopID();
		}

		m_SerializedObject->ApplyModifiedProperties();

		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::CenteredButton("   Add Component   "))
		{
			ComponentPicker::Open();
		}

		TypeId componentType;
		if (ComponentPicker::GetResult(&componentType))
		{
			const ClassInfo* info = ClassDB::GetInfo(componentType);
			for (Object* target : m_SerializedObject->GetTargets())
			{
				m_AddedComponents.push_back(std::make_pair(static_cast<Entity*>(target), static_cast<Component*>(info->Create())));
			}
		}

		bool hasAddedComponents = m_AddedComponents.size() > 0, hasRemovedComponents = m_RemovedComponents.size() > 0;
		if (hasAddedComponents)
		{
			for (auto& pair : m_AddedComponents)
			{
				EditorObjectManager::AddComponent(pair.first, pair.second);
			}
			m_AddedComponents.clear();
		}
		if (hasRemovedComponents)
		{
			for (Component* component : m_RemovedComponents)
			{
				EditorObjectManager::RemoveComponent(component);
			}
			m_RemovedComponents.clear();
		}
		if (hasAddedComponents || hasRemovedComponents)
		{
			OnDisable();
			OnEnable();
		}
	}

	void EntityEditor::OnEntityDestroy()
	{
		if (!m_SerializedObject->IsValid())
		{
			Selection::SetActiveObject(nullptr);
		}
	}
}