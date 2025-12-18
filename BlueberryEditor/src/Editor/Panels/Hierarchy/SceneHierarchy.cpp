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

	HierarchyUpdateEvent SceneHierarchy::s_HierarchyUpdated = {};

	SceneHierarchy::SceneHierarchy()
	{
		EditorObjectManager::GetEntityCreated().AddCallback<SceneHierarchy, &SceneHierarchy::Invalidate>(this);
		EditorObjectManager::GetEntityDestroyed().AddCallback<SceneHierarchy, &SceneHierarchy::Invalidate>(this);
	}

	SceneHierarchy::~SceneHierarchy()
	{
		EditorObjectManager::GetEntityCreated().RemoveCallback<SceneHierarchy, &SceneHierarchy::Invalidate>(this);
		EditorObjectManager::GetEntityDestroyed().RemoveCallback<SceneHierarchy, &SceneHierarchy::Invalidate>(this);
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
			
			ImGui::SetNextItemOpen(m_ExpandedNodes.count(id));
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

			DrawNode(nodes, i);

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
							Invalidate();
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

		if (!isHoveringAny && ImGui::IsWindowHovered())
		{
			if (ImGui::IsMouseClicked(0) && !ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
			{
				Selection::SetActiveObject(nullptr);
			}
			else if (ImGui::IsMouseClicked(1))
			{
				ImGui::OpenPopup(ImGui::GetID("Popup"));
			}
		}

		if (ImGui::BeginPopup(ImGui::GetID("Popup")))
		{
			DrawCreateEntity();
			ImGui::EndPopup();
		}

		if (scene != nullptr && !m_IsValid)
		{
			UpdateTree();
			m_IsValid = true;
		}
	}

	HierarchyUpdateEvent& SceneHierarchy::GetHierarchyUpdated()
	{
		return s_HierarchyUpdated;
	}

	void SceneHierarchy::DrawNode(List<TransformTreeNode>& nodes, const size_t& index)
	{
		ImGui::PushID(index);
		TransformTreeNode& node = nodes[index];
		Entity* entity = node.entity.Get();
		if (!ImGui::IsItemToggledOpen())
		{
			ImGuiID id = ImGui::GetID(0);
			static ImGuiID activeId = 0;
			if (ImGui::IsItemClicked(0))
			{
				activeId = id;
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
								for (; j < to - 1; ++j)
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
					if (!Selection::IsActiveObject(entity))
					{
						Selection::SetActiveObject(entity);
					}
				}
				m_LastClickedItem = index;
			}
			if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0) && index == m_LastClickedItem)
			{
				m_RenamingEntity = entity;	// check IsItemActiveAsInputText
			}
			else if (ImGui::IsItemHovered() && ImGui::IsMouseReleased(0) && activeId == id)
			{
				activeId = 0;
				if (!ImGui::IsKeyDown(ImGuiKey_LeftCtrl) && !ImGui::IsKeyDown(ImGuiKey_LeftShift))
				{
					Selection::SetActiveObject(entity);
				}
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
					Transform* parent = entity->GetTransform();
					for (Object* object : Selection::GetActiveObjects())
					{
						if (object->GetType() == Entity::Type)
						{
							Entity* selectedEntity = static_cast<Entity*>(object);
							if (entity != selectedEntity)
							{
								selectedEntity->GetTransform()->SetParent(parent);
							}
						}
					}
					Invalidate();
				}
			}
			ImGui::EndDragDropTarget();
		}
		
		if (ImGui::BeginPopup(ImGui::GetID("Popup")))
		{
			DrawCreateEntity();
			DrawCloneEntity();
			DrawUnpackPrefabEntity();
			DrawDestroyEntity();
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
				bool isActive = entity->IsActiveInHierarchy();
				static ImVec4 colors[3] = { ImVec4(1.0f, 1.0f, 1.0f, 1.0f), ImVec4(0.0f, 1.0f, 0.0f, 1.0f), ImVec4(0.0f, 0.7f, 0.0f, 1.0f) };
				ImVec4 color = colors[node.type];
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

	void SceneHierarchy::DrawCreateEntity()
	{
		if (ImGui::MenuItem("Create Empty Entity"))
		{
			Entity* entity = EditorObjectManager::CreateEntity("Empty Entity");
			m_RenamingEntity = entity;

			Object* selectedObject = Selection::GetActiveObject();
			if (selectedObject != nullptr && selectedObject->IsClassType(Entity::Type))
			{
				Entity* selectedEntity = static_cast<Entity*>(selectedObject);
				entity->GetTransform()->SetParent(selectedEntity->GetTransform());
				m_ExpandedNodes.insert(selectedEntity->GetObjectId());
			}
		}
	}

	void SceneHierarchy::DrawCloneEntity()
	{
		if (ImGui::MenuItem("Clone Entity"))
		{
			Object* selectedObject = Selection::GetActiveObject();
			if (selectedObject != nullptr && selectedObject->IsClassType(Entity::Type))
			{
				Entity* entity = static_cast<Entity*>(selectedObject);
				Entity* newEntity = EditorObjectManager::CloneEntity(entity);
				newEntity->GetTransform()->SetParent(entity->GetTransform()->GetParent());
			}
		}
	}

	void SceneHierarchy::DrawDestroyEntity()
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
		}
	}

	void SceneHierarchy::DrawUnpackPrefabEntity()
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
					Invalidate();
				}
			}
		}
	}

	void SceneHierarchy::Invalidate()
	{
		m_IsValid = false;
	}

	void SceneHierarchy::UpdateTree()
	{
		m_TransformTree.Update(m_CurrentScene == nullptr ? List<ObjectPtr<Entity>>() : m_CurrentScene->GetRootEntities());
		s_HierarchyUpdated.Invoke();
	}
}