#include "Blueberry\Scene\Components\BoxCollider.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Core\ClassDB.h"

#include <Jolt\Jolt.h>
#include <Jolt\Physics\Collision\Shape\BoxShape.h>

namespace Blueberry
{
	OBJECT_DEFINITION(BoxCollider, Collider)
	{
		DEFINE_BASE_FIELDS(BoxCollider, Collider)
		DEFINE_FIELD(BoxCollider, m_Size, BindingType::Vector3, {})
	}

	const Vector3& BoxCollider::GetSize()
	{
		return m_Size;
	}

	JPH::Shape* BoxCollider::GetShape()
	{
		Transform* transform = GetTransform();
		Vector3 scale = transform->GetLocalScale();
		return new JPH::BoxShape(JPH::RVec3(m_Size.x * scale.x, m_Size.y * scale.y, m_Size.z * scale.z));
	}
}