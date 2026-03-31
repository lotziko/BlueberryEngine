#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Entity;
	class Transform;

	struct TransformTreeNode
	{
		ObjectPtr<Entity> entity;
		ObjectPtr<Transform> transform;
		uint32_t type;
		int depth = 0;
	};

	class TransformTree
	{
	public:
		TransformTree() = default;

		List<TransformTreeNode>& GetNodes();
		void Update(const List<ObjectPtr<Entity>>& roots);

	private:
		void Populate(Transform* parent, int depth);

	private:
		List<TransformTreeNode> m_Nodes;
	};
}