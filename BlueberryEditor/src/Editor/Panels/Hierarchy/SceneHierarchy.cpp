#include "SceneHierarchy.h"

#include "Blueberry\Core\Base.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Core\ObjectCloner.h"

#include "Editor\Selection.h"
#include "Editor\EditorSceneManager.h"
#include "Editor\Prefabs\PrefabManager.h"
#include "Editor\Prefabs\PrefabInstance.h"
#include "Editor\Menu\EditorMenuManager.h"
#include "Editor\Misc\ImGuiHelper.h"
#include "Editor\Misc\EntityHelper.h"
#include "Editor\Undo\UndoManager.h"
#include "Editor\Events\SceneEvents.h"

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
		SceneEvents::GetSceneLoaded().AddCallback<SceneHierarchy, &SceneHierarchy::Invalidate>(this);
		SceneEvents::GetEntityCreated().AddCallback<SceneHierarchy, &SceneHierarchy::Invalidate>(this);
		SceneEvents::GetEntityDestroyed().AddCallback<SceneHierarchy, &SceneHierarchy::Invalidate>(this);
		SceneEvents::GetHierarchyUpdated().AddCallback<SceneHierarchy, &SceneHierarchy::Invalidate>(this);
		Selection::GetSelectionChanged().AddCallback<SceneHierarchy, &SceneHierarchy::OnSelectionChange>(this);
	}

	SceneHierarchy::~SceneHierarchy()
	{
		SceneEvents::GetSceneLoaded().RemoveCallback<SceneHierarchy, &SceneHierarchy::Invalidate>(this);
		SceneEvents::GetEntityCreated().RemoveCallback<SceneHierarchy, &SceneHierarchy::Invalidate>(this);
		SceneEvents::GetEntityDestroyed().RemoveCallback<SceneHierarchy, &SceneHierarchy::Invalidate>(this);
		SceneEvents::GetHierarchyUpdated().RemoveCallback<SceneHierarchy, &SceneHierarchy::Invalidate>(this);
		Selection::GetSelectionChanged().RemoveCallback<SceneHierarchy, &SceneHierarchy::OnSelectionChange>(this);
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
			bool isActive = Selection::IsActiveObject(entity);
			ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_AllowOverlap | (isActive ? ImGuiTreeNodeFlags_Selected : 0) | (hasChildren ? ImGuiTreeNodeFlags_OpenOnArrow : (ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen));
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

			if (m_ScrollRequest == i)
			{
				ImGui::ScrollToItem(ImGuiScrollFlags_KeepVisibleEdgeY);
				m_ScrollRequest = UINT64_MAX;
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
							Entity* selectedEntity = static_cast<Entity*>(object);
							Transform* transform = selectedEntity->GetTransform();
							Transform* parent = entity->GetTransform()->GetParent();
							size_t index = entity->GetTransform()->GetSiblingIndex() + 1;
							if (transform->GetParent() == parent)
							{
								size_t currentIndex = transform->GetSiblingIndex();
								if (currentIndex < index)
								{
									--index;
								}
								UndoManager::SetTransformParent(transform, parent, index);
							}
							else
							{
								UndoManager::SetTransformParent(transform, parent, index);
							}
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

		if (m_DestroyedEntities.size() > 0)
		{
			for (ObjectId id : m_DestroyedEntities)
			{
				UndoManager::DestroyObject(ObjectDB::GetObject(id));
			}
			m_DestroyedEntities.clear();
			Invalidate();
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

	void SceneHierarchy::DrawNode(List<TransformTreeNode>& nodes, size_t index)
	{
		ImGui::PushID(static_cast<int>(index));
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
				//else
				//{
				//	if (!Selection::IsActiveObject(entity))
				//	{
				//		//Selection::SetActiveObject(entity);
				//	}
				//}
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
			List<ObjectId> ids;
			ids.push_back(entity->GetObjectId());
			for (size_t i = 0; i < entity->GetComponentCount(); ++i)
			{
				ids.push_back(entity->GetComponentAt(i)->GetObjectId());
			}
			ImGui::SetDragDropPayload("OBJECT_ID", ids.data(), ids.size() * sizeof(ObjectId));
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
					List<Object*> objectsToReparent;
					if (Selection::IsActiveObject(object))
					{
						objectsToReparent = Selection::GetActiveObjects();
					}
					else
					{
						objectsToReparent.push_back(object);
					}
					for (Object* object : objectsToReparent)
					{
						if (object->GetType() == Entity::Type)
						{
							Entity* selectedEntity = static_cast<Entity*>(object);
							if (entity != selectedEntity)
							{
								Transform* transform = selectedEntity->GetTransform();
								UndoManager::SetTransformParent(transform, parent, parent->GetChildrenCount());
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
				static ImVec4 colors[3] = { ImGui::GetStyleColorVec4(ImGuiCol_Text), ImGui::GetStyleColorVec4(ImGuiCol_SeparatorActive), ImGui::GetStyleColorVec4(ImGuiCol_SeparatorActive) };
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
		if (ImGui::MenuItem("Create Empty"))
		{
			Object* selectedObject = Selection::GetActiveObject();
			if (selectedObject != nullptr && selectedObject->IsClassType(Entity::Type))
			{
				Entity* selectedEntity = static_cast<Entity*>(selectedObject);
				Entity* entity = EntityHelper::Create("Empty Entity");
				UndoManager::AddEntity(entity);
				UndoManager::SetTransformParent(entity->GetTransform(), selectedEntity->GetTransform());
				m_RenamingEntity = entity;
				m_ExpandedNodes.insert(selectedEntity->GetObjectId());
			}
		}
		if (ImGui::MenuItem("Create Empty Parent"))
		{
			Object* selectedObject = Selection::GetActiveObject();
			if (selectedObject != nullptr && selectedObject->IsClassType(Entity::Type))
			{
				Entity* selectedEntity = static_cast<Entity*>(selectedObject);
				if (!PrefabManager::IsPartOfPrefabInstance(selectedEntity))
				{
					Entity* entity = EntityHelper::Create("Empty Entity");
					UndoManager::AddEntity(entity);
					UndoManager::SetTransformParent(entity->GetTransform(), selectedEntity->GetTransform()->GetParent(), selectedEntity->GetTransform()->GetSiblingIndex());
					UndoManager::SetTransformParent(selectedEntity->GetTransform(), entity->GetTransform());
					m_RenamingEntity = entity;
					m_ExpandedNodes.insert(entity->GetObjectId());
				}
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
				Entity* newEntity = nullptr;
				if (PrefabManager::IsPartOfPrefabInstance(entity))
				{
					if (PrefabManager::IsPrefabInstanceRoot(entity))
					{
						PrefabInstance* newInstance = PrefabManager::CloneInstance(PrefabManager::GetInstance(entity));
						newEntity = newInstance->GetEntity();
					}
				}
				else
				{
					newEntity = static_cast<Entity*>(ObjectCloner::Clone(entity));
				}
				Transform* newTransform = newEntity->GetTransform();
				Transform* transform = entity->GetTransform();
				UndoManager::AddEntity(newEntity);
				UndoManager::SetTransformParent(newTransform, transform->GetParent(), false, transform->GetSiblingIndex() + 1);
				Selection::SetActiveObject(newEntity);
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
					m_DestroyedEntities.push_back(object->GetObjectId());
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

	void SceneHierarchy::OnSelectionChange()
	{
		for (Object* object : Selection::GetActiveObjects())
		{
			if (object->IsClassType(Entity::Type))
			{
				Entity* entity = static_cast<Entity*>(object);
				ObjectId id = entity->GetObjectId();
				if (entity->GetScene() == m_CurrentScene)
				{
					List<TransformTreeNode>& nodes = m_TransformTree.GetNodes();
					size_t size = nodes.size();
					for (size_t i = 0; i < size; ++i)
					{
						const TransformTreeNode& node = nodes[i];
						if (node.entity == entity)
						{
							int currentDepth = node.depth;
							m_ScrollRequest = i;
							for (size_t j = i; j-- > 0;)
							{
								const TransformTreeNode& currentNode = nodes[j];
								int depth = currentNode.depth;
								if (depth < currentDepth)
								{
									m_ExpandedNodes.insert(currentNode.entity.Get()->GetObjectId());
									currentDepth = depth;
								}
								if (depth == 0)
								{
									break;
								}
							}
							break;
						}
					}
				}
			}
		}
	}
}