#include "MeshRendererEditor.h"

#include "Blueberry\Scene\Components\MeshRenderer.h"

#include "Editor\Gizmos\Gizmos.h"

namespace Blueberry
{
	void MeshRendererEditor::OnDrawSceneSelected()
	{
		/*for (Object* target : m_SerializedObject->GetTargets())
		{
			MeshRenderer* renderer = static_cast<MeshRenderer*>(target);
			AABB bounds = renderer->GetBounds();
			Gizmos::SetMatrix(Matrix::Identity);
			Gizmos::DrawBox(bounds.Center, static_cast<Vector3>(bounds.Extents) * 2);
		}*/
	}
}
