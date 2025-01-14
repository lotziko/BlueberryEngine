#pragma once

#include "Octree.h"

namespace Blueberry
{
	class Renderer;

	class RendererTree
	{
	public:
		static void Add(const ObjectId& id, const AABB& bounds);
		static void Remove(const ObjectId& id, const AABB& bounds);
		static void Update(const ObjectId& id, const AABB& previousBounds, const AABB& newBounds);
		static void Cull(DirectX::XMVECTOR* planes, std::vector<ObjectId>& result);
		static void GatherBounds(std::vector<AABB>& result);

	private:
		static Octree m_Octree;
	};
}