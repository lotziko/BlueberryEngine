#include "Blueberry\Scene\Components\Renderer.h"

#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Renderer, Component)
	{
		DEFINE_BASE_FIELDS(Renderer, Component)
		DEFINE_FIELD(Renderer, m_IsCastingShadows, BindingType::Bool, FieldOptions())
	}

	int Renderer::GetSortingOrder() const
	{
		return m_SortingOrder;
	}

	void Renderer::SetSortingOrder(int sortingOrder)
	{
		m_SortingOrder = sortingOrder;
	}

	bool Renderer::IsCastingShadows() const
	{
		return m_IsCastingShadows;
	}

	void Renderer::SetCastingShadows(bool castingShadows)
	{
		m_IsCastingShadows = castingShadows;
	}
}