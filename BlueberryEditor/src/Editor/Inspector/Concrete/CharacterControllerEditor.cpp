#include "CharacterControllerEditor.h"

#include "Blueberry\Scene\Components\CharacterController.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"

#include "Editor\Gizmos\Gizmos.h"

namespace Blueberry
{
	void CharacterControllerEditor::OnDrawSceneSelected()
	{
		for (Object* target : m_SerializedObject->GetTargets())
		{
			auto controller = static_cast<CharacterController*>(target);
			auto transform = controller->GetTransform();

			float height = controller->GetHeight();
			float radius = controller->GetRadius();

			Gizmos::SetColor(Color(0, 1, 0, 1));
			Gizmos::SetMatrix(transform->GetLocalToWorldMatrix());
			Gizmos::DrawCapsule(Vector3(0, height / 2 + radius, 0), height, radius);
		}
	}
}
