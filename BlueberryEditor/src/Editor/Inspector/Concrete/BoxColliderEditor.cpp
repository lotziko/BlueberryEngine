#include "BoxColliderEditor.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\BoxCollider.h"

#include "Editor\Gizmos\Gizmos.h"

namespace Blueberry
{
	void BoxColliderEditor::OnDrawSceneSelected()
	{
		for (Object* target : m_SerializedObject->GetTargets())
		{
			auto collider = static_cast<BoxCollider*>(target);
			auto transform = collider->GetTransform();
			Gizmos::SetMatrix(transform->GetLocalToWorldMatrix());
			Gizmos::SetColor(Color(0, 1, 0, 1));
			Gizmos::DrawBox(Vector3::Zero, collider->GetSize() * 2);
		}
	}
}
