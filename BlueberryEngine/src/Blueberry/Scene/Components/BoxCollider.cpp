#include "bbpch.h"
#include "BoxCollider.h"

#include "Blueberry\Scene\Entity.h"

#include <Jolt\Jolt.h>
#include <Jolt\Physics\Collision\Shape\BoxShape.h>

namespace Blueberry
{
	OBJECT_DEFINITION(Collider, BoxCollider)

	const Vector3& BoxCollider::GetSize()
	{
		return m_Size;
	}

	void BoxCollider::BindProperties()
	{
		BEGIN_OBJECT_BINDING(BoxCollider)
		BIND_FIELD(FieldInfo(TO_STRING(m_Entity), &BoxCollider::m_Entity, BindingType::ObjectPtr).SetObjectType(Entity::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_Size), &BoxCollider::m_Size, BindingType::Vector3))
		END_OBJECT_BINDING()
	}

	JPH::Shape* BoxCollider::GetShape()
	{
		return new JPH::BoxShape(JPH::RVec3(m_Size.x, m_Size.y, m_Size.z));
	}
}