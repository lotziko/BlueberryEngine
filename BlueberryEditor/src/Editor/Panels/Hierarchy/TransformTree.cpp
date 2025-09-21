#include "TransformTree.h"

#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Entity.h"

namespace Blueberry
{
	List<TransformTreeNode>& TransformTree::GetNodes()
	{
		return m_Nodes;
	}

	void TransformTree::Update(const List<ObjectPtr<Entity>>& roots)
	{
		m_Nodes.clear();
		for (auto& root : roots)
		{
			Transform* transform = root.Get()->GetTransform();
			TransformTreeNode node = {};
			node.entity = root;
			node.transform = transform;
			m_Nodes.push_back(std::move(node));
			Populate(transform, 1);
		}
	}

	void TransformTree::Populate(Transform* parent, const int& depth)
	{
		for (auto& child : parent->GetChildren())
		{
			Entity* entity = child.Get()->GetEntity();;
			TransformTreeNode node = {};
			node.entity = entity;
			node.transform = child;
			node.depth = depth;
			m_Nodes.push_back(std::move(node));
			Populate(child.Get(), depth + 1);
		}
	}
}
