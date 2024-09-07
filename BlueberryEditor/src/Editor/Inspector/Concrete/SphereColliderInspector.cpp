#include "bbpch.h"
#include "SphereColliderInspector.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\SphereCollider.h"

#include "Editor\Gizmos\Gizmos.h"

namespace Blueberry
{
	void SphereColliderInspector::DrawScene(Object* object)
	{
		auto collider = static_cast<SphereCollider*>(object);
		auto transform = collider->GetEntity()->GetTransform();
		Gizmos::SetMatrix(transform->GetLocalToWorldMatrix());
		Gizmos::SetColor(Color(0, 1, 0, 1));
		Gizmos::DrawCircle(Vector3::Zero, collider->GetRadius());
	}
}
