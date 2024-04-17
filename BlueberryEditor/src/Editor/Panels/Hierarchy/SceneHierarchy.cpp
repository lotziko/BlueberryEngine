#include "bbpch.h"
#include "SceneHierarchy.h"

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Scene.h"

#include "Editor\Selection.h"
#include "Editor\EditorSceneManager.h"

#include "imgui\imgui.h"

namespace Blueberry
{
	void SceneHierarchy::DrawUI()
	{
		ImGui::Begin("Hierarchy");

		Scene* scene = EditorSceneManager::GetScene();

		if (scene != nullptr)
		{
			std::vector<ObjectPtr<Entity>> entities = std::vector<ObjectPtr<Entity>>(scene->GetEntities());

			for (auto entity : entities)
			{
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

		ImGuiTreeNodeFlags flags = ((Selection::GetActiveObject() == entity) ? ImGuiTreeNodeFlags_Selected : 0) | (entity->GetTransform()->GetChildrenCount() > 0 ? ImGuiTreeNodeFlags_OpenOnArrow : ImGuiTreeNodeFlags_Leaf);
		flags |= ImGuiTreeNodeFlags_SpanAvailWidth;

		bool opened = ImGui::TreeNodeEx((void*)entity, flags, entity->GetName().c_str());
		if (ImGui::IsItemClicked())
		{
			Selection::SetActiveObject(entity);
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
				Object* item = ObjectDB::GetObject(*id);

				if (item != nullptr && item != entity && ImGui::AcceptDragDropPayload("OBJECT_ID"))
				{
					((Entity*)item)->GetTransform()->SetParent(entity->GetTransform());
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

		/*ImGui::SameLine();
		if (Selection::GetActiveObject() == entity)
		{
			std::string buffer = entity->GetName();
			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
			if (ImGui::InputText("###rename", buffer.data(), buffer.size(), ImGuiInputTextFlags_AutoSelectAll))
			{

			}
			ImGui::PopStyleVar();
		}
		else
		{
			ImGui::Text("%s", entity->GetName().c_str());
		}*/

		if (opened)
		{
			if (entity != nullptr)
			{
				for (auto child : entity->GetTransform()->GetChildren())
				{
					DrawEntity(child->GetEntity());
				}
			}
			//ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
			//bool opened = ImGui::TreeNodeEx((void*)9817239, flags, entity->ToString().c_str());
			//if (opened)
			//	ImGui::TreePop();
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