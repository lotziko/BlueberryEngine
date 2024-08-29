#include "bbpch.h"
#include "SphereCollider.h"

#include "Blueberry\Scene\Entity.h"

#include <Jolt\Jolt.h>
#include <Jolt\Physics\Collision\Shape\SphereShape.h>

namespace Blueberry
{
	OBJECT_DEFINITION(Collider, SphereCollider)

	void SphereCollider::BindProperties()
	{
		BEGIN_OBJECT_BINDING(SphereCollider)
		BIND_FIELD(FieldInfo(TO_STRING(m_Entity), &SphereCollider::m_Entity, BindingType::ObjectPtr).SetObjectType(Entity::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_Radius), &SphereCollider::m_Radius, BindingType::Float))
		END_OBJECT_BINDING()
	}

	JPH::Shape* SphereCollider::GetShape()
	{
		return new JPH::SphereShape(m_Radius);
	}
}