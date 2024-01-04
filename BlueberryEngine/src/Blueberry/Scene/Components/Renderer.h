#pragma once

#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	class Renderer : public Component
	{
		OBJECT_DECLARATION(Renderer)

	public:
		static void BindProperties();

		const int& GetSortingOrder();
		void SetSortingOrder(const int& sortingOrder);

	protected:
		int m_SortingOrder = 0;
	};
}