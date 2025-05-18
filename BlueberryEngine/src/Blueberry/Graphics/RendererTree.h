#pragma once

#include "Blueberry\Core\Object.h"
#include "Octree.h"

namespace Blueberry
{
	class Renderer;

	class RendererTree
	{
	public:
		void Add(const ObjectId& id, const AABB& bounds);
		void Remove(const ObjectId& id, const AABB& bounds);
		void Update(const ObjectId& id, const AABB& previousBounds, const AABB& newBounds);
		void Cull(DirectX::XMVECTOR* planes, List<ObjectId>& result);
		void GatherBounds(List<AABB>& result);

	private:
		Octree m_Octree = Octree(Vector3::Zero, 10.0f, 1.0f, 1.0f);
	};
}