#include "Blueberry\Scene\Components\ProbeVolume.h"

#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(ProbeVolume, Component)
	{
		DEFINE_BASE_FIELDS(ProbeVolume, Component)
		DEFINE_FIELD(ProbeVolume, m_Bounds, BindingType::AABB, FieldOptions())
		DEFINE_FIELD(ProbeVolume, m_Size, BindingType::Vector3Int, FieldOptions())
		DEFINE_ITERATOR(ProbeVolume)
		DEFINE_EXECUTE_ALWAYS()
	}

	const AABB& ProbeVolume::GetBounds()
	{
		return m_Bounds;
	}

	void ProbeVolume::SetBounds(const AABB& bounds)
	{
		m_Bounds = bounds;
	}

	const Vector3Int& ProbeVolume::GetSize()
	{
		return m_Size;
	}

	void ProbeVolume::SetSize(const Vector3Int& size)
	{
		m_Size = size;
	}
}