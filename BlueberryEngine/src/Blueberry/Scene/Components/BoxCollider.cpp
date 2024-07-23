#include "bbpch.h"
#include "BoxCollider.h"

#include "Blueberry\Scene\Entity.h"

#include <Jolt\Jolt.h>
#include <Jolt\Physics\Collision\Shape\BoxShape.h>

namespace Blueberry
{
	OBJECT_DEFINITION(Collider, BoxCollider)

	void BoxCollider::BindProperties()
	{
		BEGIN_OBJECT_BINDING(BoxCollider)
		BIND_FIELD(FieldInfo(TO_STRING(m_Entity), &BoxCollider::m_Entity, BindingType::ObjectPtr).SetObjectType(Entity::Type))
		END_OBJECT_BINDING()
	}

	JPH::Shape* BoxCollider::GetShape()
	{
		return new JPH::BoxShape(JPH::RVec3(1.0f, 1.0f, 1.0f));
	}
}