#pragma once

#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	class BB_API Renderer : public Component
	{
		OBJECT_DECLARATION(Renderer)

	public:
		virtual const AABB& GetBounds();

		const int& GetSortingOrder();
		void SetSortingOrder(const int& sortingOrder);

	protected:
		int m_SortingOrder = 0;
	};
}