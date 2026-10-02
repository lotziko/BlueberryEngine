#pragma once

#include "Blueberry\Scene\Components\Component.h"
#include "Blueberry\Graphics\OctreeObjectInterface.h"

namespace Blueberry
{
	class BB_API Renderer : public Component, public OctreeObjectInterface
	{
		OBJECT_DECLARATION(Renderer)

	public:
		virtual const AABB& GetBounds() = 0;
		virtual const Matrix& GetLocalToWorldMatrix() = 0;

		int GetSortingOrder() const;
		void SetSortingOrder(int sortingOrder);

		bool IsCastingShadows() const;
		void SetCastingShadows(bool castingShadows);

	protected:
		virtual ObjectId GetOctreeObjectId() const override;
		virtual OctreeNode* GetOctreeNode() const override;
		virtual void SetOctreeNode(OctreeNode* node) override;

	protected:
		int m_SortingOrder = 0;
		bool m_IsCastingShadows = true;
		OctreeNode* m_OctreeNode = nullptr;
	};
}