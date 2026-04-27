#include "ProbeVolumeEditor.h"

#include "Blueberry\Scene\Components\ProbeVolume.h"

#include "Editor\Gizmos\Gizmos.h"

namespace Blueberry
{
	void ProbeVolumeEditor::OnDrawSceneSelected()
	{
		for (Object* target : m_SerializedObject->GetTargets())
		{
			ProbeVolume* volume = static_cast<ProbeVolume*>(target);
			Transform* transform = volume->GetTransform();
			AABB bounds = volume->GetBounds();

			if (bounds.Extents.x > 0)
			{
				Gizmos::SetMatrix(Matrix::Identity);
				Gizmos::DrawBox(bounds.Center, static_cast<Vector3>(bounds.Extents) * 2);
			}
		}
	}
}
