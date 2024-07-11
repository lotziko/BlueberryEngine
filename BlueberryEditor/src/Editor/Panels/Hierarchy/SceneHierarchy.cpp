#include "bbpch.h"
#include "SceneHierarchy.h"

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Scene.h"

#include "Editor\Selection.h"
#include "Editor\EditorSceneManager.h"
#include "Editor\Prefabs\PrefabManager.h"

#include "imgui\imgui.h"

namespace Blueberry
{
	void SceneHierarchy::DrawUI()
	{
		ImGui::Begin("Hierarchy");

		Scene* scene = EditorSceneManager::GetScene();

		if (scene != nullptr)
		{
			std::map<ObjectId, ObjectPtr<Entity>> entities = std::map<ObjectId, ObjectPtr<Entity>>(scene->GetEntities());

			for (auto& pair : entities)
			{
				auto entity = pair.second;
				if (entity.IsValid() && entity.Get()->GetTransform() != nullptr && entity.Get()->GetTransform()->GetParent() == nullptr)
				{
					DrawEntity(entity.Get());
				}
			}

			if (ImGui::IsMouseDown(0) && ImGui::IsWindowHovered())
			{
				Selection::SetActiveObject(nullptr);
			}

			if (ImGui::BeginPopupContextWindow(0, 1, false))
			{
				DrawCreateEntity();
				ImGui::EndPopup();
			}
		}

		ImGui::End();
	}

	void SceneHierarchy::DrawEntity(Entity* entity)
	{
		if (entity == nullptr)
		{
			return;
		}

		bool isEntityStartedRenaming = false;

		ImGuiTreeNodeFlags flags = ((Selection::IsActiveObject(entity)) ? ImGuiTreeNodeFlags_Selected : 0) | (entity->GetTransform()->GetChildrenCount() > 0 ? ImGuiTreeNodeFlags_OpenOnArrow : ImGuiTreeNodeFlags_Leaf);
		if (m_ActiveEntity != entity)
		{
			flags |= ImGuiTreeNodeFlags_SpanAvailWidth;
		}

		bool opened = ImGui::TreeNodeEx((void*)entity, flags, "");
		if (!ImGui::IsItemToggledOpen())
		{
			if (ImGui::IsItemClicked())
			{
				Selection::SetActiveObject(entity);
			}

			if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0) && Selection::IsActiveObject(entity))
			{
				m_ActiveEntity = entity;
				isEntityStartedRenaming = true;
			}
		}

		if (ImGui::BeginDragDropSource())
		{
			ObjectId id = entity->GetObjectId();
			ImGui::SetDragDropPayload("OBJECT_ID", &id, sizeof(ObjectId));
			ImGui::Text("%s", entity->GetName().c_str());
			ImGui::EndDragDropSource();
		}

		if (ImGui::BeginDragDropTarget())
		{
			const ImGuiPayload* payload = ImGui::GetDragDropPayload();
			if (payload != nullptr && payload->IsDataType("OBJECT_ID"))
			{
				ObjectId* id = (ObjectId*)payload->Data;
				Object* object = ObjectDB::GetObject(*id);

				if (object != nullptr && object != entity && ImGui::AcceptDragDropPayload("OBJECT_ID"))
				{
					((Entity*)object)->GetTransform()->SetParent(entity->GetTransform());
				}
			}
			ImGui::EndDragDropTarget();
		}

		if (ImGui::BeginPopupContextItem())
		{
			DrawCreateEntity();
			DrawDestroyEntity(entity);
			DrawRenameEntity(entity);
			ImGui::EndPopup();
		}

		if (entity != nullptr)
		{
			ImGui::SameLine();
			if (m_ActiveEntity == entity)
			{
				static char buf[256];

				if (isEntityStartedRenaming)
				{
					std::string name = entity->GetName();
					strncpy(buf, name.c_str(), sizeof(buf) - 1);
					ImGui::SetKeyboardFocusHere();
				}

				std::string name = entity->GetName();
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
				ImGui::InputText("###rename", buf, 256, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);

				if (ImGui::IsItemDeactivated())
				{
					m_ActiveEntity = nullptr;
					entity->SetName(buf);
				}

				ImGui::PopStyleVar();
			}
			else
			{
				bool isPrefab = PrefabManager::IsPrefabInstace(entity);
				if (isPrefab)
				{
					ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 255, 0, 255));
				}
				ImGui::Text("%s", entity->GetName().c_str());
				if (isPrefab)
				{
					ImGui::PopStyleColor();
				}
			}
		}

		if (opened)
		{
			if (entity != nullptr)
			{
				for (auto child : entity->GetTransform()->GetChildren())
				{
					DrawEntity(child->GetEntity());
				}
			}
			ImGui::TreePop();
		}
	}

	void SceneHierarchy::DrawCreateEntity()
	{
		if (ImGui::MenuItem("Create Empty Entity"))
		{
			Entity* entity = EditorSceneManager::GetScene()->CreateEntity("Empty Entity");

			Object* selectedObject = Selection::GetActiveObject();
			if (selectedObject != nullptr && selectedObject->IsClassType(Entity::Type))
			{
				entity->GetTransform()->SetParent(static_cast<Entity*>(selectedObject)->GetTransform());
			}
		}
	}

	void SceneHierarchy::DrawDestroyEntity(Entity*& entity)
	{
		if (ImGui::MenuItem("Delete Entity"))
		{
			EditorSceneManager::GetScene()->DestroyEntity(entity);
			if (Selection::GetActiveObject() == entity)
			{
				Selection::SetActiveObject(nullptr);
			}
			entity = nullptr;
		}
	}

	void SceneHierarchy::DrawRenameEntity(Entity* entity)
	{
		if (ImGui::MenuItem("Rename Entity"))
		{

		}
	}
}