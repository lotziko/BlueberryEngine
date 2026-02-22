#include "SkinnedMeshRendererEditor.h"

#include "Blueberry\Scene\Components\SkinnedMeshRenderer.h"

#include "Editor\Gizmos\Gizmos.h"

namespace Blueberry
{
	void SkinnedMeshRendererEditor::OnDrawSceneSelected()
	{
		for (Object* target : m_SerializedObject->GetTargets())
		{
			SkinnedMeshRenderer* renderer = static_cast<SkinnedMeshRenderer*>(target);
			AABB bounds = renderer->GetBounds();
			Gizmos::SetMatrix(Matrix::Identity);
			Gizmos::DrawBox(bounds.Center, static_cast<Vector3>(bounds.Extents) * 2);
		}
	}
}
