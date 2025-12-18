#include "Blueberry\Scene\Components\ProbeVolume.h"

#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(ProbeVolume, Component)
	{
		DEFINE_BASE_FIELDS(ProbeVolume, Component)
		DEFINE_ITERATOR(ProbeVolume)
	}

	const AABB& ProbeVolume::GetBounds()
	{
		return m_Bounds;
	}

	const Vector3Int& ProbeVolume::GetSize()
	{
		return m_Size;
	}
}