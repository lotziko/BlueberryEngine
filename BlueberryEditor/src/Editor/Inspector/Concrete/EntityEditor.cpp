#include "EntityEditor.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Component.h"

#include "Blueberry\Core\ClassDB.h"

#include "Editor\Inspector\ObjectEditorDB.h"
#include "Editor\Panels\Inspector\InspectorExpandedItemsCache.h"
#include "Editor\Panels\Scene\SceneArea.h"
#include "Editor\Misc\ImGuiHelper.h"
#include "Editor\EditorObjectManager.h"
#include "Editor\Selection.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	bool ShouldHideType(const ClassInfo& info)
	{
		if (info.createInstance == nullptr)
		{
			return true;
		}

		if (!ClassDB::IsParent(info.id, Component::Type))
		{
			return true;
		}

		return false;
	}

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
					m_ComponentsEditors.push_back(std::make_pair(list[0]->GetObjectId(), editor));
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
		ImGui::BeginPaddedArea(ImVec2(10, 5), ImVec2(10, 5));
		ImGui::Property(&m_IsActiveProperty, "Is Active");
		ImGui::EndPaddedArea();

		for (auto& pair : m_ComponentsEditors)
		{
			ImGui::PushID(pair.first);

			String name = ObjectDB::GetObject(pair.first)->GetTypeName();
			ImGui::SetNextItemOpen(InspectorExpandedItemsCache::Get(name));
			bool opened = ImGui::CollapsingHeader(name.c_str());
			if (ImGui::IsItemToggledOpen())
			{
				InspectorExpandedItemsCache::Set(name, opened);
			}

			if (ImGui::BeginPopupContextItem())
			{
				if (ImGui::MenuItem("Remove component"))
				{
					for (Object* target : pair.second->GetSerializedObject()->GetTargets())
					{
						m_RemovedComponents.push_back(static_cast<Component*>(target));
					}
				}
				ImGui::EndPopup();
			}

			if (opened)
			{
				ImGui::BeginPaddedArea(ImVec2(10, 5), ImVec2(10, 5));
				pair.second->DrawInspector();
				ImGui::EndPaddedArea();
			}

			ImGui::PopID();
		}

		m_SerializedObject->ApplyModifiedProperties();

		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::CenteredButton("   Add Component   "))
		{
			ImGui::OpenPopup("addComponent");
		}

		if (ImGui::BeginPopup("addComponent"))
		{
			auto& infos = ClassDB::GetInfos();

			for (auto& info : infos)
			{
				if (ShouldHideType(info))
				{
					continue;
				}

				if (ImGui::Selectable(info.name.c_str()))
				{
					for (Object* target : m_SerializedObject->GetTargets())
					{
						m_AddedComponents.push_back(std::make_pair(static_cast<Entity*>(target), static_cast<Component*>(info.Create())));
					}
				}
			}

			ImGui::EndPopup();
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