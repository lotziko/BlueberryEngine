#include "bbpch.h"
#include "SceneHierarchy.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Scene.h"
#include "imgui\imgui.h"

SceneHierarchy::SceneHierarchy(const Ref<Scene>& scene) : m_Scene(scene)
{
}

void SceneHierarchy::DrawUI()
{
	ImGui::Begin("Hierarchy");

	for (auto entity : m_Scene->GetEntities())
	{
		DrawEntity(entity);
	}

	ImGui::End();
}

void SceneHierarchy::DrawEntity(const Ref<Entity>& entity)
{
	ImGui::TreeNode(entity->ToString().c_str());
}
