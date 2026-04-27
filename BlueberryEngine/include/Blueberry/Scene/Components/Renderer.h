#pragma once

#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	class BB_API Renderer : public Component
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
		int m_SortingOrder = 0;
		bool m_IsCastingShadows = true;
	};
}