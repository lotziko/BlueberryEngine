#pragma once

namespace Blueberry
{
	class OctreeNode
	{
	public:
		OctreeNode(const Vector3& center, const float& size, const float& minNodeSize, const float& looseness);

		bool Add(const AABB& bounds, const ObjectId& object);
		bool Remove(const ObjectId& object);
		bool Remove(const AABB& bounds, const ObjectId& object);
		const uint32_t GetBestFit(const Vector3& center);

		void Cull(DirectX::XMVECTOR* planes, std::vector<ObjectId>& result, bool skipChecks);
		void GatherChildrenBounds(std::vector<AABB>& result);

		std::shared_ptr<OctreeNode> ShrinkIfPossible(const float& minSize);

	private:
		void FillData(const Vector3& center, const float& size, const float& minNodeSize, const float& looseness);
		static bool Encapsulates(const AABB& first, const AABB& second);
		void SubAdd(const AABB& bounds, const ObjectId& object);
		bool SubRemove(const AABB& bounds, const ObjectId& object);
		void Split();
		bool ShouldMerge();
		void Merge();
		bool HasAnyObjects();

	private:
		Vector3 m_Center;
		AABB m_Bounds;
		float m_Size;
		float m_AdjustedSize;
		float m_MinNodeSize;
		float m_Looseness;
		std::vector<std::pair<AABB, ObjectId>> m_Objects;
		std::array<AABB, 8> m_ChildBounds;
		std::array<std::shared_ptr<OctreeNode>, 8> m_Children;

		friend class Octree;
	};

	class Octree
	{
	public:
		Octree(const Vector3& initialPosition, const float& initialSize, const float& minNodeSize, const float& looseness);

		void Add(const AABB& bounds, const ObjectId& object);
		bool Remove(const ObjectId& object);
		bool Remove(const AABB& bounds, const ObjectId& object);

		void Cull(DirectX::XMVECTOR* planes, std::vector<ObjectId>& result);
		void GatherChildrenBounds(std::vector<AABB>& result);

	private:
		void Grow(const Vector3& direction);
		void Shrink();

	private:
		std::shared_ptr<OctreeNode> m_Root;
		float m_InitialSize;
		float m_MinNodeSize;
		float m_Looseness;
	};
}