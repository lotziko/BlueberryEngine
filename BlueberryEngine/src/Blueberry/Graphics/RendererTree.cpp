#include "bbpch.h"
#include "RendererTree.h"

namespace Blueberry
{
	void RendererTree::Add(const ObjectId& id, const AABB& bounds)
	{
		m_Octree.Add(bounds, id);
	}

	void RendererTree::Remove(const ObjectId& id, const AABB& bounds)
	{
		m_Octree.Remove(bounds, id);
	}

	void RendererTree::Update(const ObjectId& id, const AABB& previousBounds, const AABB& newBounds)
	{
		m_Octree.Remove(previousBounds, id);
		m_Octree.Add(newBounds, id);
	}

	void RendererTree::Cull(DirectX::XMVECTOR* planes, List<ObjectId>& result)
	{
		m_Octree.Cull(planes, result);
	}

	void RendererTree::GatherBounds(List<AABB>& result)
	{
		m_Octree.GatherChildrenBounds(result);
	}
}
