#pragma once

#include "Editor\Panels\EditorWindow.h"
#include "TransformTree.h"

namespace Blueberry
{
	class Scene;
	class Entity;

	class SceneHierarchy : public EditorWindow
	{
		OBJECT_DECLARATION(SceneHierarchy)

	public:
		SceneHierarchy() = default;
		virtual ~SceneHierarchy() = default;

		static void Open();

		virtual void OnDrawUI() final;

	private:
		void DrawNode(List<TransformTreeNode>& nodes, const size_t& index, bool& isValid);
		void DrawCreateEntity(TransformTreeNode& node, bool& isValid);
		void DrawDestroyEntity(bool& isValid);
		void DrawUnpackPrefabEntity(bool& isValid);

		void UpdateTree();

		Scene* m_CurrentScene;
		Entity* m_RenamingEntity;
		TransformTree m_TransformTree;
		size_t m_LastClickedItem = UINT64_MAX;
		HashSet<ObjectId> m_ExpandedNodes;
	};
}