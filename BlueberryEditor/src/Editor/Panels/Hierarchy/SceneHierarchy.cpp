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

		for (auto entity : m_Scene->GetEntities())
		{
			DrawEntity(entity.get());
		}

		if (ImGui::IsMouseDown(0) && ImGui::IsWindowHovered())
		{
			Selection::SetActiveObject(nullptr);
		}

		if (ImGui::BeginPopupContextWindow(0, 1, false))
		{
			if (ImGui::MenuItem("Create Empty Entity"))
			{
				m_Scene->CreateEntity("Empty Entity");
			}
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

		ImGuiTreeNodeFlags flags = ((Selection::GetActiveObject() == entity) ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_Leaf;// | ImGuiTreeNodeFlags_OpenOnArrow;
		flags |= ImGuiTreeNodeFlags_SpanAvailWidth;

		bool opened = ImGui::TreeNodeEx((void*)entity, flags, entity->ToString().c_str());
		if (ImGui::IsItemClicked())
		{
			Selection::SetActiveObject(entity);
		}

		if (ImGui::BeginPopupContextItem())
		{
			if (ImGui::MenuItem("Delete Entity"))
			{
				m_Scene->DestroyEntity(entity);
				if (Selection::GetActiveObject() == entity)
				{
					Selection::SetActiveObject(nullptr);
				}
			}

			ImGui::EndPopup();
		}

		if (opened)
		{
			//ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
			//bool opened = ImGui::TreeNodeEx((void*)9817239, flags, entity->ToString().c_str());
			//if (opened)
			//	ImGui::TreePop();
			ImGui::TreePop();
		}
	}
}