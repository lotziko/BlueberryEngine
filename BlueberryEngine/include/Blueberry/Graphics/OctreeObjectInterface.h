#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class OctreeNode;

	class BB_API OctreeObjectInterface
	{
	public:
		virtual ~OctreeObjectInterface() = default;

	protected:
		virtual ObjectId GetOctreeObjectId() const = 0;
		virtual OctreeNode* GetOctreeNode() const = 0;
		virtual void SetOctreeNode(OctreeNode* node) = 0;

		friend class Octree;
		friend class OctreeNode;
	};
}