#include "bbpch.h"
#include "Renderer.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Component, Renderer)

	void Renderer::BindProperties()
	{
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