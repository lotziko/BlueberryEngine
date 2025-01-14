#include "bbpch.h"
#include "Octree.h"

namespace Blueberry
{
	const int MAX_OBJECTS_COUNT = 8;

	OctreeNode::OctreeNode(const Vector3& center, const float& size, const float& minNodeSize, const float& looseness)
	{
		FillData(center, size, minNodeSize, looseness);
	}

	bool OctreeNode::Add(const AABB& bounds, const ObjectId& object)
	{
		if (!Encapsulates(m_Bounds, bounds))
		{
			return false;
		}
		SubAdd(bounds, object);
		return true;
	}

	bool OctreeNode::Remove(const ObjectId& object)
	{
		bool removed = false;
		for (int i = 0; i < m_Objects.size(); ++i)
		{
			if (m_Objects[i].second == object)
			{
				m_Objects.erase(m_Objects.begin() + i);
				removed = true;
				break;
			}
		}

		if (!removed && m_Children[0])
		{
			for (int i = 0; i < 8; ++i) 
			{
				removed = m_Children[i]->Remove(object);
				if (removed)
				{
					break;
				}
			}
		}

		if (removed && m_Children[0])
		{
			if (ShouldMerge())
			{
				Merge();
			}
		}

		return removed;
	}

	bool OctreeNode::Remove(const AABB& bounds, const ObjectId& object)
	{
		if (!Encapsulates(m_Bounds, bounds))
		{
			return false;
		}
		return SubRemove(bounds, object);
	}

	const uint32_t OctreeNode::GetBestFit(const Vector3& center)
	{
		return (center.x <= m_Center.x ? 0 : 1) + (center.y >= m_Center.y ? 0 : 4) + (center.z <= m_Center.z ? 0 : 2);
	}

	void OctreeNode::Cull(DirectX::XMVECTOR* planes, std::vector<ObjectId>& result, bool skipChecks)
	{
		if (skipChecks)
		{
			for (int i = 0; i < m_Objects.size(); ++i)
			{
				result.emplace_back(m_Objects[i].second);
			}
			if (m_Children[0])
			{
				for (int i = 0; i < 8; ++i)
				{
					m_Children[i]->Cull(planes, result, true);
				}
			}
		}
		else
		{
			for (int i = 0; i < m_Objects.size(); ++i)
			{
				if (m_Objects[i].first.ContainedBy(planes[0], planes[1], planes[2], planes[3], planes[4], planes[5]))
				{
					result.emplace_back(m_Objects[i].second);
				}
			}
			if (m_Children[0])
			{
				for (int i = 0; i < 8; ++i)
				{
					DirectX::ContainmentType type = m_ChildBounds[i].ContainedBy(planes[0], planes[1], planes[2], planes[3], planes[4], planes[5]);
					if (type == DirectX::ContainmentType::CONTAINS)
					{
						m_Children[i]->Cull(planes, result, true);
					}
					else if (type == DirectX::ContainmentType::INTERSECTS)
					{
						m_Children[i]->Cull(planes, result, false);
					}
				}
			}
		}
	}

	void OctreeNode::GatherChildrenBounds(std::vector<AABB>& result)
	{
		if (m_Children[0])
		{
			for (int i = 0; i < 8; ++i)
			{
				result.emplace_back(m_ChildBounds[i]);
				m_Children[i]->GatherChildrenBounds(result);
			}
		}
	}

	std::shared_ptr<OctreeNode> OctreeNode::ShrinkIfPossible(const float& minSize)
	{
		if (m_Size < (2 * minSize))
		{
			return nullptr;
		}
		if (m_Objects.size() == 0 && !m_Children[0])
		{
			return nullptr;
		}
		uint32_t bestFit = MAXUINT32;
		for (int i = 0; i < m_Objects.size(); ++i)
		{
			std::pair<AABB, ObjectId> object = m_Objects[i];
			int newBestFit = GetBestFit(object.first.Center);
			if (i == 0 || newBestFit == bestFit)
			{
				if (Encapsulates(m_ChildBounds[newBestFit], object.first))
				{
					if (bestFit == MAXUINT32)
					{
						bestFit = newBestFit;
					}
				}
				else
				{
					return nullptr;
				}
			}
			else
			{
				return nullptr;
			}
		}

		if (m_Children[0])
		{
			bool childHadContent = false;
			for (int i = 0; i < 8; ++i)
			{
				if (m_Children[i]->HasAnyObjects())
				{
					if (childHadContent)
					{
						return nullptr;
					}
					if (bestFit != MAXUINT32 && bestFit != i)
					{
						return nullptr;
					}
					childHadContent = true;
					bestFit = i;
				}
			}
		}

		if (!m_Children[0])
		{
			FillData(m_ChildBounds[bestFit].Center, m_Size / 2, minSize, m_Looseness);
			return nullptr;
		}

		if (bestFit == MAXUINT32)
		{
			return nullptr;
		}

		return m_Children[bestFit];
	}

	void OctreeNode::FillData(const Vector3& center, const float& size, const float& minNodeSize, const float& looseness)
	{
		m_Center = center;
		m_Size = size;
		m_AdjustedSize = size * looseness;
		m_Bounds = AABB(m_Center, Vector3(m_AdjustedSize, m_AdjustedSize, m_AdjustedSize) * 0.5f);
		m_MinNodeSize = minNodeSize;
		m_Looseness = looseness;

		float quarter = size / 4.0f;
		float half = size / 2.0f * looseness;
		Vector3 childExtents = Vector3(half, half, half) * 0.5f;

		m_ChildBounds[0] = AABB(m_Center + Vector3(-quarter, quarter, -quarter), childExtents);
		m_ChildBounds[1] = AABB(m_Center + Vector3(quarter, quarter, -quarter), childExtents);
		m_ChildBounds[2] = AABB(m_Center + Vector3(-quarter, quarter, quarter), childExtents);
		m_ChildBounds[3] = AABB(m_Center + Vector3(quarter, quarter, quarter), childExtents);
		m_ChildBounds[4] = AABB(m_Center + Vector3(-quarter, -quarter, -quarter), childExtents);
		m_ChildBounds[5] = AABB(m_Center + Vector3(quarter, -quarter, -quarter), childExtents);
		m_ChildBounds[6] = AABB(m_Center + Vector3(-quarter, -quarter, quarter), childExtents);
		m_ChildBounds[7] = AABB(m_Center + Vector3(quarter, -quarter, quarter), childExtents);
	}

	bool OctreeNode::Encapsulates(const AABB& first, const AABB& second)
	{
		return first.Contains(second) == DirectX::ContainmentType::CONTAINS;
	}

	void OctreeNode::SubAdd(const AABB& bounds, const ObjectId& object)
	{
		if (!m_Children[0])
		{
			if (m_Objects.size() < MAX_OBJECTS_COUNT || (m_Size / 2.0f) < m_MinNodeSize)
			{
				m_Objects.emplace_back(std::make_pair(bounds, object));
				return;
			}

			uint32_t bestFitChild;
			if (!m_Children[0])
			{
				Split();
				if (!m_Children[0])
				{
					BB_ERROR("Child creating failed.");
					return;
				}

				for (int i = m_Objects.size() - 1; i >= 0; i--)
				{
					std::pair<AABB, ObjectId> object = m_Objects[i];
					AABB objectBounds = object.first;
					bestFitChild = GetBestFit(objectBounds.Center);
					if (Encapsulates(m_Children[bestFitChild]->m_Bounds, objectBounds))
					{
						m_Children[bestFitChild]->SubAdd(object.first, object.second);
						m_Objects.erase(m_Objects.begin() + i);
					}
				}
			}
		}

		uint32_t bestFit = GetBestFit(bounds.Center);
		if (Encapsulates(m_Children[bestFit]->m_Bounds, bounds))
		{
			m_Children[bestFit]->SubAdd(bounds, object);
		}
		else
		{
			m_Objects.emplace_back(std::make_pair(bounds, object));
		}
	}

	bool OctreeNode::SubRemove(const AABB& bounds, const ObjectId& object)
	{
		bool removed = false;

		for (uint32_t i = 0; i < m_Objects.size(); ++i)
		{
			if (m_Objects[i].second == object)
			{
				m_Objects.erase(m_Objects.begin() + i);
				removed = true;
				break;
			}
		}

		if (!removed && m_Children[0])
		{
			uint32_t bestFit = GetBestFit(bounds.Center);
			removed = m_Children[bestFit]->SubRemove(bounds, object);
		}

		if (removed && m_Children[0])
		{
			if (ShouldMerge())
			{
				Merge();
			}
		}

		if (removed == false)
		{
			BB_INFO("cant");
		}

		return removed;
	}

	void OctreeNode::Split()
	{
		float quarter = m_Size / 4.0f;
		float newSize = m_Size / 2.0f;
		m_Children[0] = std::make_shared<OctreeNode>(m_Center + Vector3(-quarter, quarter, -quarter), newSize, m_MinNodeSize, m_Looseness);
		m_Children[1] = std::make_shared<OctreeNode>(m_Center + Vector3(quarter, quarter, -quarter), newSize, m_MinNodeSize, m_Looseness);
		m_Children[2] = std::make_shared<OctreeNode>(m_Center + Vector3(-quarter, quarter, quarter), newSize, m_MinNodeSize, m_Looseness);
		m_Children[3] = std::make_shared<OctreeNode>(m_Center + Vector3(quarter, quarter, quarter), newSize, m_MinNodeSize, m_Looseness);
		m_Children[4] = std::make_shared<OctreeNode>(m_Center + Vector3(-quarter, -quarter, -quarter), newSize, m_MinNodeSize, m_Looseness);
		m_Children[5] = std::make_shared<OctreeNode>(m_Center + Vector3(quarter, -quarter, -quarter), newSize, m_MinNodeSize, m_Looseness);
		m_Children[6] = std::make_shared<OctreeNode>(m_Center + Vector3(-quarter, -quarter, quarter), newSize, m_MinNodeSize, m_Looseness);
		m_Children[7] = std::make_shared<OctreeNode>(m_Center + Vector3(quarter, -quarter, quarter), newSize, m_MinNodeSize, m_Looseness);
	}

	bool OctreeNode::ShouldMerge()
	{
		uint32_t totalObjects = m_Objects.size();
		if (m_Children[0])
		{
			for (uint32_t i = 0; i < 8; ++i)
			{
				OctreeNode* child = m_Children[i].get();
				if (child->m_Children[0])
				{
					return false;
				}
				totalObjects += child->m_Objects.size();
			}
		}
		return totalObjects <= MAX_OBJECTS_COUNT;
	}

	void OctreeNode::Merge()
	{
		for (uint32_t i = 0; i < 8; ++i)
		{
			OctreeNode* child = m_Children[i].get();
			int numObjects = child->m_Objects.size();
			for (int j = numObjects - 1; j >= 0; --j)
			{
				m_Objects.emplace_back(child->m_Objects[j]);
			}
			m_Children[i] = nullptr;
		}
	}

	bool OctreeNode::HasAnyObjects()
	{
		if (m_Objects.size() > 0)
		{
			return true;
		}

		if (m_Children[0])
		{
			for (int i = 0; i < 8; i++) 
			{
				if (m_Children[i]->HasAnyObjects())
				{
					return true;
				}
			}
		}

		return false;
	}

	Octree::Octree(const Vector3& initialPosition, const float& initialSize, const float& minNodeSize, const float& looseness)
	{
		m_Looseness = std::clamp(looseness, 1.0f, 2.0f);
		m_Root = std::make_shared<OctreeNode>(initialPosition, initialSize, minNodeSize, m_Looseness);
		m_InitialSize = initialSize;
		m_MinNodeSize = minNodeSize;
	}

	void Octree::Add(const AABB& bounds, const ObjectId& object)
	{
		int count = 0;
		while (!m_Root->Add(bounds, object))
		{
			Grow(bounds.Center - m_Root->m_Center);
			if (++count > 10)
			{
				BB_ERROR("Can't grow the octree more.");
				break;
			}
		}
	}

	bool Octree::Remove(const ObjectId& object)
	{
		bool removed = m_Root->Remove(object);

		if (removed)
		{
			Shrink();
		}

		return removed;
	}

	bool Octree::Remove(const AABB& bounds, const ObjectId& object)
	{
		bool removed = m_Root->Remove(bounds, object);

		if (removed)
		{
			Shrink();
		}

		return removed;
	}

	void Octree::Cull(DirectX::XMVECTOR* planes, std::vector<ObjectId>& result)
	{
		m_Root->Cull(planes, result, false);
	}

	void Octree::GatherChildrenBounds(std::vector<AABB>& result)
	{
		m_Root->GatherChildrenBounds(result);
	}

	void Octree::Grow(const Vector3& direction)
	{
		int xDirection = direction.x >= 0 ? 1 : -1;
		int yDirection = direction.y >= 0 ? 1 : -1;
		int zDirection = direction.z >= 0 ? 1 : -1;
		std::shared_ptr<OctreeNode> oldRoot = m_Root;
		Vector3 oldCenter = oldRoot->m_Center;
		float oldSize = oldRoot->m_Size;
		float half = oldSize / 2.0f;
		float newSize = oldSize * 2.0f;
		Vector3 newCenter = oldCenter + Vector3(xDirection * half, yDirection * half, zDirection * half);

		m_Root = std::make_shared<OctreeNode>(newCenter, newSize, m_MinNodeSize, m_Looseness);

		if (oldRoot->HasAnyObjects())
		{
			uint32_t rootPos = m_Root->GetBestFit(oldCenter);
			for (int i = 0; i < 8; ++i)
			{
				if (i == rootPos)
				{
					m_Root->m_Children[i] = oldRoot;
				}
				else
				{
					xDirection = i % 2 == 0 ? -1 : 1;
					yDirection = i > 3 ? -1 : 1;
					zDirection = (i < 2 || (i > 3 && i < 6)) ? -1 : 1;
					m_Root->m_Children[i] = std::make_shared<OctreeNode>(newCenter + Vector3(xDirection * half, yDirection * half, zDirection * half), oldSize, m_MinNodeSize, m_Looseness);
				}
			}
		}
	}

	void Octree::Shrink()
	{
		std::shared_ptr<OctreeNode> newRoot = m_Root->ShrinkIfPossible(m_InitialSize);
		if (newRoot != nullptr)
		{
			m_Root = newRoot;
		}
	}
}