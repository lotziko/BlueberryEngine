#include "SphereColliderEditor.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\SphereCollider.h"

#include "Editor\Gizmos\Gizmos.h"

namespace Blueberry
{
	void SphereColliderEditor::OnDrawSceneSelected()
	{
		for (Object* target : m_SerializedObject->GetTargets())
		{
			auto collider = static_cast<SphereCollider*>(target);
			auto transform = collider->GetTransform();
			Gizmos::SetMatrix(transform->GetLocalToWorldMatrix());
			Gizmos::SetColor(Color(0, 1, 0, 1));
			Gizmos::DrawSphere(Vector3::Zero, collider->GetRadius());
		}
	}
}
