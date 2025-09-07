#include "MeshRendererEditor.h"

#include "Blueberry\Scene\Components\MeshRenderer.h"

#include "Editor\Gizmos\Gizmos.h"

namespace Blueberry
{
	void MeshRendererEditor::OnDrawScene()
	{
		MeshRenderer* renderer = static_cast<MeshRenderer*>(m_Object);
		AABB bounds = renderer->GetBounds();
		Gizmos::SetMatrix(Matrix::Identity);
		//Gizmos::DrawBox(bounds.Center, static_cast<Vector3>(bounds.Extents) * 2);
	}
}
