#pragma once

#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	class Renderer : public Component
	{
		OBJECT_DECLARATION(Renderer)

	public:
		virtual const AABB& GetBounds();

		const int& GetSortingOrder();
		void SetSortingOrder(const int& sortingOrder);

		static void BindProperties();

	protected:
		int m_SortingOrder = 0;
	};
}