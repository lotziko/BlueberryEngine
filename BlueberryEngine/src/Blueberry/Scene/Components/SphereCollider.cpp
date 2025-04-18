#include "bbpch.h"
#include "SphereCollider.h"

#include "Blueberry\Scene\Entity.h"

#include <Jolt\Jolt.h>
#include <Jolt\Physics\Collision\Shape\SphereShape.h>

namespace Blueberry
{
	OBJECT_DEFINITION(SphereCollider, Collider)
	{
		DEFINE_BASE_FIELDS(SphereCollider, Collider)
		DEFINE_FIELD(SphereCollider, m_Radius, BindingType::Float, {})
	}

	const float& SphereCollider::GetRadius()
	{
		return m_Radius;
	}

	JPH::Shape* SphereCollider::GetShape()
	{
		return new JPH::SphereShape(m_Radius);
	}
}