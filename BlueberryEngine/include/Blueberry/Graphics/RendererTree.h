#pragma once

#include "Blueberry\Core\Object.h"
#include "Octree.h"

namespace Blueberry
{
	class Renderer;

	class RendererTree
	{
	public:
		void Add(ObjectId id, const AABB& bounds);
		void Remove(ObjectId id, const AABB& bounds);
		void Update(ObjectId id, const AABB& previousBounds, const AABB& newBounds);
		void Cull(DirectX::XMVECTOR* planes, List<ObjectId>& result);
		void GatherBounds(List<AABB>& result);

	private:
		Octree m_Octree = Octree(Vector3::Zero, 10.0f, 1.0f, 1.0f);
	};
}