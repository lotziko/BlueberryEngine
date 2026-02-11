#pragma once

#include "Editor\Panels\EditorWindow.h"
#include "TransformTree.h"

#include "Blueberry\Events\Event.h"

namespace Blueberry
{
	class Scene;
	class Entity;

	using HierarchyUpdateEvent = Event<>;

	class SceneHierarchy : public EditorWindow
	{
		OBJECT_DECLARATION(SceneHierarchy)

	public:
		SceneHierarchy();
		virtual ~SceneHierarchy();

		static void Open();

		virtual void OnDrawUI() final;

		static HierarchyUpdateEvent& GetHierarchyUpdated();

	private:
		void DrawNode(List<TransformTreeNode>& nodes, const size_t& index);
		void DrawCreateEntity();
		void DrawCloneEntity();
		void DrawDestroyEntity();
		void DrawUnpackPrefabEntity();

		void Invalidate();
		void UpdateTree();

		void OnSelectionChange();

		Scene* m_CurrentScene;
		Entity* m_RenamingEntity;
		TransformTree m_TransformTree;
		size_t m_LastClickedItem = UINT64_MAX;
		HashSet<ObjectId> m_ExpandedNodes;
		List<ObjectId> m_DestroyedEntities;
		bool m_IsValid = false;
		size_t m_ScrollRequest = UINT64_MAX;

		static HierarchyUpdateEvent s_HierarchyUpdated;
	};
}