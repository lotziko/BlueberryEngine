#include "bbpch.h"
#include "BoxColliderInspector.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\BoxCollider.h"

#include "Editor\Gizmos\Gizmos.h"

namespace Blueberry
{
	void BoxColliderInspector::DrawScene(Object* object)
	{
		auto collider = static_cast<BoxCollider*>(object);
		auto transform = collider->GetTransform();
		Gizmos::SetMatrix(transform->GetLocalToWorldMatrix());
		Gizmos::SetColor(Color(0, 1, 0, 1));
		Gizmos::DrawBox(Vector3::Zero, collider->GetSize() * 2);
	}
}
