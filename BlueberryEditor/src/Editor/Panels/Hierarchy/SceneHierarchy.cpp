#include "SceneHierarchy.h"

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Core\ClassDB.h"

#include "Editor\Selection.h"
#include "Editor\EditorSceneManager.h"
#include "Editor\EditorObjectManager.h"
#include "Editor\Prefabs\PrefabManager.h"
#include "Editor\Menu\EditorMenuManager.h"
#include "Editor\Misc\ImGuiHelper.h"

#include <imgui\imgui_internal.h>

namespace Blueberry
{
	OBJECT_DEFINITION(SceneHierarchy, EditorWindow)
	{
		DEFINE_BASE_FIELDS(SceneHierarchy, EditorWindow)
		EditorMenuManager::AddItem("Window/Hierarchy", &SceneHierarchy::Open);
	}

	void SceneHierarchy::Open()
	{
		EditorWindow* window = GetWindow(SceneHierarchy::Type);
		window->SetTitle("Hierarchy");
		window->Show();
	}

	void SceneHierarchy::OnDrawUI()
	{
		Scene* scene = EditorSceneManager::GetScene();

		if (scene != m_CurrentScene)
		{
			m_CurrentScene = scene;
			UpdateTree();
		}

		List<TransformTreeNode>& nodes = m_TransformTree.GetNodes();
		size_t size = nodes.size();
		bool isValid = true;
		bool isHoveringAny = false;
		int currentDepth = 0;

		ImVec2 spacing = ImGui::GetStyle().ItemSpacing;
		ImVec2 padding = ImGui::GetStyle().FramePadding;
		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(spacing.x, 0));
		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(padding.x, spacing.y / 2));
		for (size_t i = 0; i < size; ++i)
		{
			TransformTreeNode& node = nodes[i];
			if (node.depth > currentDepth || !node.entity.IsValid())
			{
				continue;
			}
			while (currentDepth >= node.depth + 1)
			{
				ImGui::TreePop();
				--currentDepth;
			}

			Entity* entity = node.entity.Get();
			ObjectId id = entity->GetObjectId();
			bool hasChildren = (i < size - 1 && nodes[i + 1].depth > node.depth);
			ImGuiTreeNodeFlags flags = (Selection::IsActiveObject(entity) ? ImGuiTreeNodeFlags_Selected : 0) | (hasChildren ? ImGuiTreeNodeFlags_OpenOnArrow : (ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen));
			flags |= ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_FramePadding;
			
			ImGui::SetNextItemOpen(m_ExpandedNodes.count(id));//m_ExpandedNodes
			bool opened = ImGui::TreeNodeEx(entity, flags, "");
			if (ImGui::IsItemToggledOpen())
			{
				if (opened)
				{
					m_ExpandedNodes.insert(id);
				}
				else
				{
					m_ExpandedNodes.erase(id);
				}
			}

			if (ImGui::IsItemHovered())
			{
				isHoveringAny = true;
			}

			DrawNode(nodes, i, isValid);

			ImVec2 pos = ImGui::GetCursorScreenPos();
			if (ImGui::IsDragDropActive())
			{
				ImGui::Dummy(ImVec2(ImGui::GetContentRegionAvail().x, spacing.y));

				if (ImGui::BeginDragDropTarget())
				{
					const ImGuiPayload* payload = ImGui::GetDragDropPayload();
					if (payload != nullptr && payload->IsDataType("OBJECT_ID"))
					{
						ObjectId* id = static_cast<ObjectId*>(payload->Data);
						Object* object = ObjectDB::GetObject(*id);

						if (ImGui::AcceptDragDropPayload("OBJECT_ID"))
						{
							Transform* transform = (static_cast<Entity*>(object))->GetTransform();
							transform->SetParent(entity->GetTransform()->GetParent());
							transform->SetSiblingIndex(entity->GetTransform()->GetSiblingIndex() + 1);
							isValid = false;
						}
					}
					ImGui::EndDragDropTarget();
				}
			}
			ImGui::SetCursorScreenPos(pos);

			if (hasChildren && opened) 
			{
				++currentDepth;
			}
		}
		ImGui::PopStyleVar(2);

		while (currentDepth > 0)
		{
			ImGui::TreePop();
			--currentDepth;
		}

		if (!isHoveringAny && ImGui::IsMouseClicked(0) && ImGui::IsWindowHovered() && !ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
		{
			Selection::SetActiveObject(nullptr);
		}

		if (!isValid)
		{
			UpdateTree();
		}
	}

	void SceneHierarchy::DrawNode(List<TransformTreeNode>& nodes, const size_t& index, bool& isValid)
	{
		ImGui::PushID(index);
		TransformTreeNode& node = nodes[index];
		Entity* entity = node.entity.Get();
		if (!ImGui::IsItemToggledOpen())
		{
			if (ImGui::IsItemClicked(0))
			{
				if (ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
				{
					Selection::AddActiveObject(entity);
				}
				else if (ImGui::IsKeyDown(ImGuiKey_LeftShift))
				{
					if (m_LastClickedItem != UINT64_MAX && index != m_LastClickedItem)
					{
						size_t from = std::min(index, m_LastClickedItem);
						size_t to = std::max(index, m_LastClickedItem);
						for (size_t j = from; j <= to; ++j)
						{
							TransformTreeNode& nodeToSelect = nodes[j];
							Entity* entityToSelect = nodeToSelect.entity.Get();
							Selection::AddActiveObject(entityToSelect);
							if (m_ExpandedNodes.count(entityToSelect->GetObjectId()) == 0)
							{
								for (j += 1; j < to - 1; ++j)
								{
									if (nodes[j + 1].depth == nodeToSelect.depth)
									{
										break;
									}
								}
							}
						}
					}
				}
				else
				{
					if (index == m_LastClickedItem)
					{
						if (ImGui::IsMouseDoubleClicked(0))
						{
							m_RenamingEntity = entity;
						}
					}
					else
					{
						Selection::SetActiveObject(entity);
					}
				}
				m_LastClickedItem = index;
			}
			else if (ImGui::IsItemClicked(1))
			{
				if (!Selection::IsActiveObject(entity))
				{
					Selection::SetActiveObject(entity);
				}
				ImGui::OpenPopup(ImGui::GetID("Popup"));
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
				ObjectId* id = static_cast<ObjectId*>(payload->Data);
				Object* object = ObjectDB::GetObject(*id);

				if (object != nullptr && object != entity && ImGui::AcceptDragDropPayload("OBJECT_ID"))
				{
					m_ExpandedNodes.insert(entity->GetObjectId());
					(static_cast<Entity*>(object))->GetTransform()->SetParent(entity->GetTransform());
					isValid = false;
				}
			}
			ImGui::EndDragDropTarget();
		}
		
		if (ImGui::BeginPopup(ImGui::GetID("Popup")))
		{
			DrawCreateEntity(node, isValid);
			DrawUnpackPrefabEntity(isValid);
			DrawDestroyEntity(isValid);
			ImGui::EndPopup();
		}

		if (entity != nullptr)
		{
			static bool isRenaming = false;
			static char buf[256];

			ImGui::SameLine();
			if (entity == m_RenamingEntity)
			{
				if (!isRenaming)
				{
					String name = entity->GetName();
					strncpy(buf, name.c_str(), sizeof(buf) - 1);
					ImGui::SetKeyboardFocusHere();
					isRenaming = true;
				}

				String name = entity->GetName();
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
				ImGui::InputText("###rename", buf, 256, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);

				if (ImGui::IsItemDeactivated())
				{
					m_RenamingEntity = nullptr;
					isRenaming = false;
					entity->SetName(buf);
				}

				ImGui::PopStyleVar();
			}
			else
			{
				bool isPrefab = PrefabManager::IsPartOfPrefabInstance(entity);
				bool isActive = entity->IsActiveInHierarchy();
				ImVec4 color = isPrefab ? ImVec4(0, 1, 0, 1) : ImVec4(1, 1, 1, 1);
				if (!isActive)
				{
					color.w *= 0.5f;
				}
				ImGui::PushStyleColor(ImGuiCol_Text, color);
				ImGui::Text("%s", entity->GetName().c_str());
				ImGui::PopStyleColor();
			}
		}
		ImGui::PopID();
	}

	void SceneHierarchy::DrawCreateEntity(TransformTreeNode& node, bool& isValid)
	{
		if (ImGui::MenuItem("Create Empty Entity"))
		{
			Entity* entity = EditorObjectManager::CreateEntity("Empty Entity");

			Object* selectedObject = Selection::GetActiveObject();
			if (selectedObject != nullptr && selectedObject->IsClassType(Entity::Type))
			{
				Entity* selectedEntity = static_cast<Entity*>(selectedObject);
				entity->GetTransform()->SetParent(selectedEntity->GetTransform());
				m_ExpandedNodes.insert(selectedEntity->GetObjectId());
				m_RenamingEntity = entity;
				isValid = false;
			}
		}
	}

	void SceneHierarchy::DrawDestroyEntity(bool& isValid)
	{
		if (ImGui::MenuItem("Delete Entity"))
		{
			for (Object* object : Selection::GetActiveObjects())
			{
				if (object->GetType() == Entity::Type)
				{
					Entity* entity = static_cast<Entity*>(object);
					EditorObjectManager::DestroyEntity(entity);
				}
			}
			Selection::SetActiveObject(nullptr);
			isValid = false;
		}
	}

	void SceneHierarchy::DrawUnpackPrefabEntity(bool& isValid)
	{
		Object* selectedObject = Selection::GetActiveObject();
		if (selectedObject != nullptr && selectedObject->IsClassType(Entity::Type))
		{
			Entity* selectedEntity = static_cast<Entity*>(selectedObject);
			if (PrefabManager::IsPrefabInstanceRoot(selectedEntity))
			{
				if (ImGui::MenuItem("Unpack prefab"))
				{
					PrefabManager::UnpackPrefabInstance(selectedEntity);
					isValid = false;
				}
			}
		}
	}

	void SceneHierarchy::UpdateTree()
	{
		m_TransformTree.Update(m_CurrentScene->GetRootEntities());
	}
}