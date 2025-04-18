#include "bbpch.h"
#include "Renderer.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Renderer, Component)
	{
		DEFINE_BASE_FIELDS(Renderer, Component)
	}

	const AABB& Renderer::GetBounds()
	{
		return AABB();
	}

	const int& Renderer::GetSortingOrder()
	{
		return m_SortingOrder;
	}

	void Renderer::SetSortingOrder(const int& sortingOrder)
	{
		m_SortingOrder = sortingOrder;
	}
}