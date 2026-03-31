#include "Blueberry\Scene\Components\Renderer.h"

#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Renderer, Component)
	{
		DEFINE_BASE_FIELDS(Renderer, Component)
	}

	int Renderer::GetSortingOrder() const
	{
		return m_SortingOrder;
	}

	void Renderer::SetSortingOrder(int sortingOrder)
	{
		m_SortingOrder = sortingOrder;
	}
}