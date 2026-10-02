#include "Blueberry\Graphics\Octree.h"

namespace Blueberry
{
	const int MAX_OBJECTS_COUNT = 8;

	static bool EncapsulatesBounds(const AABB& first, const AABB& second)
	{
		return first.Contains(second) == DirectX::ContainmentType::CONTAINS;
	}

	bool OctreeNode::Add(OctreeObjectInterface* object, const AABB& bounds)
	{
		if (!EncapsulatesBounds(m_Bounds, bounds))
		{
			return false;
		}
		SubAdd(object, bounds);
		return true;
	}

	bool OctreeNode::Remove(OctreeObjectInterface* object)
	{
		bool removed = false;
		if (object->GetOctreeNode() == this)
		{
			for (int i = 0; i < m_Objects.size(); ++i)
			{
				if (m_Objects[i].object == object)
				{
					object->SetOctreeNode(nullptr);
					if (i != m_Objects.size() - 1)
					{
						m_Objects[i] = m_Objects.back();
					}
					m_Objects.pop_back();
					removed = true;
					break;
				}
			}
		}
		
		if (removed)
		{
			OctreeNode* node = this;
			while (node != nullptr)
			{
				if (node->m_Children[0])
				{
					if (node->ShouldMerge())
					{
						node->Merge();
					}
				}
				node = node->m_Parent;
			}
		}

		return removed;
	}

	const uint32_t OctreeNode::GetBestFit(const Vector3& center)
	{
		return (center.x <= m_Center.x ? 0 : 1) + (center.y >= m_Center.y ? 0 : 4) + (center.z <= m_Center.z ? 0 : 2);
	}

	void OctreeNode::Cull(DirectX::XMVECTOR* planes, List<ObjectId>& result, bool skipChecks)
	{
		if (skipChecks)
		{
			for (int i = 0; i < m_Objects.size(); ++i)
			{
				result.push_back(m_Objects[i].id);
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
				if (m_Objects[i].bounds.ContainedBy(planes[0], planes[1], planes[2], planes[3], planes[4], planes[5]))
				{
					result.push_back(m_Objects[i].id);
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

	void OctreeNode::GatherChildrenBounds(List<AABB>& result)
	{
		if (m_Children[0])
		{
			for (int i = 0; i < 8; ++i)
			{
				result.push_back(m_ChildBounds[i]);
				m_Children[i]->GatherChildrenBounds(result);
			}
		}
		else
		{
			result.push_back(m_Bounds);
		}
	}

	OctreeNode* OctreeNode::ShrinkIfPossible(float minSize)
	{
		if (m_Size < (2 * minSize))
		{
			return nullptr;
		}
		if (m_Objects.size() == 0 && !m_Children[0])
		{
			return nullptr;
		}
		uint32_t bestFit = UINT32_MAX;
		for (int i = 0; i < m_Objects.size(); ++i)
		{
			ObjectData& objectData = m_Objects[i];
			int newBestFit = GetBestFit(objectData.bounds.Center);
			if (i == 0 || newBestFit == bestFit)
			{
				if (EncapsulatesBounds(m_ChildBounds[newBestFit], objectData.bounds))
				{
					if (bestFit == UINT32_MAX)
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
					if (bestFit != UINT32_MAX && bestFit != i)
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
			FillData(m_ChildBounds[bestFit].Center, m_Size / 2, m_MinNodeSize, m_Looseness);
			return nullptr;
		}

		if (bestFit == UINT32_MAX)
		{
			return nullptr;
		}

		return m_Children[bestFit];
	}

	void OctreeNode::FillData(const Vector3& center, float size, float minNodeSize, float looseness)
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

	void OctreeNode::SubAdd(OctreeObjectInterface* object, const AABB& bounds)
	{
		if (!m_Children[0])
		{
			if (m_Objects.size() < MAX_OBJECTS_COUNT || (m_Size / 2.0f) < m_MinNodeSize)
			{
				object->SetOctreeNode(this);
				m_Objects.push_back({ object->GetOctreeObjectId(), object, bounds });
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

				for (int i = static_cast<int>(m_Objects.size() - 1); i >= 0; i--)
				{
					ObjectData& objectData = m_Objects[i];
					AABB objectBounds = objectData.bounds;
					bestFitChild = GetBestFit(objectBounds.Center);
					if (EncapsulatesBounds(m_Children[bestFitChild]->m_Bounds, objectBounds))
					{
						m_Children[bestFitChild]->SubAdd(objectData.object, objectData.bounds);
						if (i != m_Objects.size() - 1)
						{
							m_Objects[i] = m_Objects.back();
						}
						m_Objects.pop_back();
					}
				}
			}
		}

		uint32_t bestFit = GetBestFit(bounds.Center);
		if (EncapsulatesBounds(m_Children[bestFit]->m_Bounds, bounds))
		{
			m_Children[bestFit]->SubAdd(object, bounds);
		}
		else
		{
			object->SetOctreeNode(this);
			m_Objects.push_back({ object->GetOctreeObjectId(), object, bounds });
		}
	}

	void OctreeNode::Split()
	{
		float quarter = m_Size / 4.0f;
		float newSize = m_Size / 2.0f;
		m_Children[0] = m_Tree->Allocate(this, m_Center + Vector3(-quarter, quarter, -quarter), newSize, m_MinNodeSize, m_Looseness);
		m_Children[1] = m_Tree->Allocate(this, m_Center + Vector3(quarter, quarter, -quarter), newSize, m_MinNodeSize, m_Looseness);
		m_Children[2] = m_Tree->Allocate(this, m_Center + Vector3(-quarter, quarter, quarter), newSize, m_MinNodeSize, m_Looseness);
		m_Children[3] = m_Tree->Allocate(this, m_Center + Vector3(quarter, quarter, quarter), newSize, m_MinNodeSize, m_Looseness);
		m_Children[4] = m_Tree->Allocate(this, m_Center + Vector3(-quarter, -quarter, -quarter), newSize, m_MinNodeSize, m_Looseness);
		m_Children[5] = m_Tree->Allocate(this, m_Center + Vector3(quarter, -quarter, -quarter), newSize, m_MinNodeSize, m_Looseness);
		m_Children[6] = m_Tree->Allocate(this, m_Center + Vector3(-quarter, -quarter, quarter), newSize, m_MinNodeSize, m_Looseness);
		m_Children[7] = m_Tree->Allocate(this, m_Center + Vector3(quarter, -quarter, quarter), newSize, m_MinNodeSize, m_Looseness);
	}

	bool OctreeNode::ShouldMerge()
	{
		uint32_t totalObjects = static_cast<uint32_t>(m_Objects.size());
		if (m_Children[0])
		{
			for (uint32_t i = 0; i < 8; ++i)
			{
				OctreeNode* child = m_Children[i];
				if (child->m_Children[0])
				{
					return false;
				}
				totalObjects += static_cast<uint32_t>(child->m_Objects.size());
			}
		}
		return totalObjects <= MAX_OBJECTS_COUNT;
	}

	void OctreeNode::Merge()
	{
		for (uint32_t i = 0; i < 8; ++i)
		{
			OctreeNode* child = m_Children[i];
			for (int j = static_cast<int>(child->m_Objects.size()) - 1; j >= 0; --j)
			{
				ObjectData& objectData = child->m_Objects[j];
				objectData.object->SetOctreeNode(this);
				m_Objects.push_back(objectData);
			}
			m_Tree->Release(child);
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

	Octree::Octree(const Vector3& initialPosition, float initialSize, float minNodeSize, float looseness)
	{
		m_Looseness = std::clamp(looseness, 1.0f, 2.0f);
		m_Root = Allocate(nullptr, initialPosition, initialSize, minNodeSize, m_Looseness);
		m_InitialSize = initialSize;
		m_MinNodeSize = minNodeSize;
	}

	void Octree::Add(OctreeObjectInterface* object, const AABB& bounds)
	{
		int count = 0;
		while (!m_Root->Add(object, bounds))
		{
			Grow(bounds.Center - m_Root->m_Center);
			if (++count > 10)
			{
				BB_ERROR("Can't grow the octree more.");
				break;
			}
		}
	}

	void Octree::Update(OctreeObjectInterface* object, const AABB& bounds)
	{
		OctreeNode* node = object->GetOctreeNode();
		if (node != nullptr && EncapsulatesBounds(node->m_Bounds, bounds))
		{
			uint32_t bestFit = node->GetBestFit(bounds.Center);
			bool canMoveDown = node->m_Children[0] && EncapsulatesBounds(node->m_ChildBounds[bestFit], bounds);

			if (!canMoveDown)
			{
				for (int i = 0; i < node->m_Objects.size(); ++i)
				{
					OctreeNode::ObjectData& objectData = node->m_Objects[i];
					if (objectData.object == object)
					{
						objectData.bounds = bounds;
						return;
					}
				}
			}
		}
		if (node != nullptr)
		{
			if (!node->Remove(object))
			{
				BB_ERROR("Octree object is missing from its owning node.");
				return;
			}
		}

		Add(object, bounds);
		Shrink();
	}

	bool Octree::Remove(OctreeObjectInterface* object)
	{
		OctreeNode* node = object->GetOctreeNode();
		if (node == nullptr)
		{
			return false;
		}
		bool removed = node->Remove(object);
		if (removed)
		{
			Shrink();
		}
		return removed;
	}

	void Octree::Cull(DirectX::XMVECTOR* planes, List<ObjectId>& result)
	{
		m_Root->Cull(planes, result, false);
	}

	void Octree::GatherChildrenBounds(List<AABB>& result)
	{
		m_Root->GatherChildrenBounds(result);
	}

	OctreeNode* Octree::Allocate(OctreeNode* parent, const Vector3& center, float size, float minNodeSize, float looseness)
	{
		OctreeNode* node;
		if (m_FreeNodes.size() > 0)
		{
			node = m_FreeNodes.back();
			m_FreeNodes.pop_back();
		}
		else
		{
			uint32_t id = m_MaxNodeId;
			uint32_t chunkIndex = id / NodesPerChunk;
			uint32_t slotIndex = id % NodesPerChunk;
			if (chunkIndex == m_Chunks.size())
			{
				m_Chunks.push_back(std::make_unique<OctreeChunk>());
			}
			node = &(*m_Chunks[chunkIndex])[slotIndex];
			node->m_Id = id;
			node->m_Tree = this;
			++m_MaxNodeId;
		}
		node->m_Parent = parent;
		node->FillData(center, size, minNodeSize, looseness);
		return node;
	}

	void Octree::Release(OctreeNode* node)
	{
		if (node == nullptr)
		{
			return;
		}
		for (OctreeNode* child : node->m_Children)
		{
			Release(child);
		}
		node->m_Objects.clear();
		node->m_Children.fill(nullptr);
		node->m_Parent = nullptr;
		m_FreeNodes.push_back(node);
	}

	void Octree::Grow(const Vector3& direction)
	{
		int xDirection = direction.x >= 0 ? 1 : -1;
		int yDirection = direction.y >= 0 ? 1 : -1;
		int zDirection = direction.z >= 0 ? 1 : -1;
		OctreeNode* oldRoot = m_Root;
		Vector3 oldCenter = oldRoot->m_Center;
		float oldSize = oldRoot->m_Size;
		float half = oldSize / 2.0f;
		float newSize = oldSize * 2.0f;
		Vector3 newCenter = oldCenter + Vector3(xDirection * half, yDirection * half, zDirection * half);

		m_Root = Allocate(nullptr, newCenter, newSize, m_MinNodeSize, m_Looseness);

		bool releaseOldRoot = true;
		if (oldRoot->HasAnyObjects())
		{
			uint32_t rootPos = m_Root->GetBestFit(oldCenter);
			for (int i = 0; i < 8; ++i)
			{
				if (i == rootPos)
				{
					m_Root->m_Children[i] = oldRoot;
					oldRoot->m_Parent = m_Root;
					releaseOldRoot = false;
				}
				else
				{
					xDirection = i % 2 == 0 ? -1 : 1;
					yDirection = i > 3 ? -1 : 1;
					zDirection = (i < 2 || (i > 3 && i < 6)) ? -1 : 1;
					m_Root->m_Children[i] = Allocate(m_Root, newCenter + Vector3(xDirection * half, yDirection * half, zDirection * half), oldSize, m_MinNodeSize, m_Looseness);
				}
			}
		}
		if (releaseOldRoot)
		{
			Release(oldRoot);
		}
	}

	void Octree::Shrink()
	{
		OctreeNode* newRoot = m_Root->ShrinkIfPossible(m_InitialSize);
		if (newRoot != nullptr)
		{
			OctreeNode* oldRoot = m_Root;
			for (int i = 0; i < 8; ++i)
			{
				OctreeNode*& child = oldRoot->m_Children[i];
				if (child == newRoot)
				{
					child = nullptr;
					break;
				}
			}
			newRoot->m_Parent = nullptr;
			m_Root = newRoot;
			Release(oldRoot);
		}
	}
}