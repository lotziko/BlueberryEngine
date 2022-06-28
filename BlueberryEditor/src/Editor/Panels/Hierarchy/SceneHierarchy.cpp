#include "bbpch.h"
#include "SceneHierarchy.h"

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Scene.h"

#include "Editor\Selection.h"

#include "imgui\imgui.h"

namespace Blueberry
{
	SceneHierarchy::SceneHierarchy(const Ref<Scene>& scene) : m_Scene(scene)
	{
	}

	void SceneHierarchy::DrawUI()
	{
		ImGui::Begin("Hierarchy");

		std::vector<Ref<Entity>> entities = std::vector<Ref<Entity>>(m_Scene->GetEntities());

		for (auto entity : entities)
		{
			if (entity != nullptr && entity->GetTransform()->GetParent() == nullptr)
			{
				DrawEntity(entity.get());
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

		bool opened = ImGui::TreeNodeEx((void*)entity, flags, entity->ToString().c_str());
		if (ImGui::IsItemClicked())
		{
			Selection::SetActiveObject(entity);
		}

		if (ImGui::BeginPopupContextItem())
		{
			DrawCreateEntity();
			DrawDestroyEntity(entity);
			ImGui::EndPopup();
		}

		if (opened)
		{
			for (auto child : entity->GetTransform()->GetChildren())
			{
				DrawEntity(child->GetEntity());
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
			Ref<Entity> entity = m_Scene->CreateEntity("Empty Entity");

			Object* selectedObject = Selection::GetActiveObject();
			if (selectedObject != nullptr && selectedObject->IsClassType(Entity::Type))
			{
				entity->GetTransform()->SetParent(static_cast<Entity*>(selectedObject)->GetTransform());
			}
		}
	}

	void SceneHierarchy::DrawDestroyEntity(Entity* entity)
	{
		if (ImGui::MenuItem("Delete Entity"))
		{
			m_Scene->DestroyEntity(entity);
			if (Selection::GetActiveObject() == entity)
			{
				Selection::SetActiveObject(nullptr);
			}
		}
	}
}