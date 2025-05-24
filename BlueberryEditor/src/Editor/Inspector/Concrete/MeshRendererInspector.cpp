#include "MeshRendererInspector.h"

#include "Blueberry\Scene\Components\MeshRenderer.h"

#include "Editor\Gizmos\Gizmos.h"

namespace Blueberry
{
	void MeshRendererInspector::DrawScene(Object* object)
	{
		MeshRenderer* renderer = static_cast<MeshRenderer*>(object);
		AABB bounds = renderer->GetBounds();
		Gizmos::SetMatrix(Matrix::Identity);
		//Gizmos::DrawBox(bounds.Center, static_cast<Vector3>(bounds.Extents) * 2);
	}
}
