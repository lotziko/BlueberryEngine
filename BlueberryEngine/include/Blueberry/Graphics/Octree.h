#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "OctreeObjectInterface.h"

namespace Blueberry
{
	class Octree;

	class OctreeNode
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		OctreeNode() = default;

		bool Add(OctreeObjectInterface* object, const AABB& bounds);
		bool Remove(OctreeObjectInterface* object);
		const uint32_t GetBestFit(const Vector3& center);

		void Cull(DirectX::XMVECTOR* planes, List<ObjectId>& result, bool skipChecks);
		void GatherChildrenBounds(List<AABB>& result);

		OctreeNode* ShrinkIfPossible(float minSize);

	private:
		void FillData(const Vector3& center, float size, float minNodeSize, float looseness);
		void SubAdd(OctreeObjectInterface* object, const AABB& bounds);
		void Split();
		bool ShouldMerge();
		void Merge();
		bool HasAnyObjects();

	private:
		struct ObjectData
		{
			ObjectId id;
			OctreeObjectInterface* object;
			AABB bounds;
		};

		Vector3 m_Center;
		AABB m_Bounds;
		float m_Size;
		float m_AdjustedSize;
		float m_MinNodeSize;
		float m_Looseness;
		List<ObjectData> m_Objects;
		Array<AABB, 8> m_ChildBounds;
		OctreeNode* m_Parent;
		Array<OctreeNode*, 8> m_Children;
		uint32_t m_Id;
		Octree* m_Tree;

		friend class Octree;
	};

	class Octree
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		Octree(const Octree&) = delete;
		Octree& operator=(const Octree&) = delete;

		Octree(const Vector3& initialPosition, float initialSize, float minNodeSize, float looseness);

		void Add(OctreeObjectInterface* object, const AABB& bounds);
		void Update(OctreeObjectInterface* object, const AABB& bounds);
		bool Remove(OctreeObjectInterface* object);

		void Cull(DirectX::XMVECTOR* planes, List<ObjectId>& result);
		void GatherChildrenBounds(List<AABB>& result);

	private:
		inline OctreeNode* GetNode(uint32_t id)
		{
			return &(*m_Chunks[id / NodesPerChunk])[id % NodesPerChunk];
		}
		OctreeNode* Allocate(OctreeNode* parent, const Vector3& center, float size, float minNodeSize, float looseness);
		void Release(OctreeNode* node);
		void Grow(const Vector3& direction);
		void Shrink();

	private:
		static constexpr uint32_t NodesPerChunk = 256;
		using OctreeChunk = Array<OctreeNode, NodesPerChunk>;
		List<std::unique_ptr<OctreeChunk>> m_Chunks;
		List<OctreeNode*> m_FreeNodes;
		uint32_t m_MaxNodeId = 0;

		OctreeNode* m_Root;
		float m_InitialSize;
		float m_MinNodeSize;
		float m_Looseness;

		friend class OctreeNode;
	};
}