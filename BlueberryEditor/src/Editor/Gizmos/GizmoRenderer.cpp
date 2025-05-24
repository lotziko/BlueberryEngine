#include "GizmoRenderer.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Component.h"
#include "Blueberry\Graphics\PerCameraDataConstantBuffer.h"

#include "Editor\Selection.h"
#include "Editor\Gizmos\Gizmos.h"
#include "Editor\Inspector\ObjectInspectorDB.h"
#include "Editor\Inspector\ObjectInspector.h"

#include "Blueberry\Graphics\RendererTree.h"

#include "Blueberry\Graphics\OpenXRRenderer.h"

#include "Blueberry\Scene\Scene.h"

namespace Blueberry
{
	void GizmoRenderer::Draw(Scene* scene, Camera* camera, GfxTexture* target)
	{
		PerCameraDataConstantBuffer::BindData(camera, target);

		// TODO add cache and draw also child entities

		//Gizmos::Begin();
		//Gizmos::SetMatrix(Matrix::Identity);
		//Gizmos::SetColor(Color(1, 0, 0, 1));

		//Frustum frustum;
		//frustum.CreateFromMatrix(frustum, OpenXRRenderer::multiviewProjectionMatrix[0], false);
		//frustum.Transform(frustum, OpenXRRenderer::multiviewViewMatrix[0].Invert());
		//Gizmos::DrawFrustum(frustum);

		//Gizmos::SetColor(Color(0, 1, 0, 1));
		//frustum = {};
		//frustum.CreateFromMatrix(frustum, OpenXRRenderer::multiviewProjectionMatrix[1], false);
		//frustum.Transform(frustum, OpenXRRenderer::multiviewViewMatrix[1].Invert());
		//Gizmos::DrawFrustum(frustum);

		////Gizmos::SetMatrix(OpenXRRenderer::multiviewViewMatrix[0].Invert());
		////Gizmos::DrawBox(Vector3::Zero, Vector3(0.1f, 0.1f, 1));
		////Gizmos::DrawBox(Vector3::Forward, Vector3(0.1f, 0.1f, 1));
		////Gizmos::SetColor(Color(0, 1, 0, 1));
		////Gizmos::SetMatrix(OpenXRRenderer::multiviewViewMatrix[1].Invert());
		////Gizmos::DrawBox(Vector3::Zero, Vector3(0.1f, 0.1f, 1));
		//Gizmos::End();

		/*Gizmos::SetMatrix(Matrix::Identity);
		Gizmos::SetColor(Color(1, 1, 1, 1));
		Gizmos::Begin();
		List<AABB> bounds = {};
		scene->GetRendererTree().GatherBounds(bounds);
		for (int i = 0; i < bounds.size(); ++i)
		{
			Gizmos::DrawBox(bounds[i].Center, (Vector3)bounds[i].Extents * 2);
		}
		Gizmos::End();*/

		Object* activeObject = Selection::GetActiveObject();
		if (activeObject != nullptr)
		{
			if (activeObject->IsClassType(Entity::Type))
			{
				Entity* entity = static_cast<Entity*>(activeObject);
				if (entity->IsActiveInHierarchy())
				{
					for (auto& component : entity->GetComponents())
					{
						ObjectInspector* inspector = ObjectInspectorDB::GetInspector(component->GetType());
						if (inspector != nullptr)
						{
							Gizmos::SetColor(Color(1, 1, 1, 1));
							Gizmos::Begin();
							inspector->DrawScene(component);
							Gizmos::End();
						}
					}
				}
			}
		}
	}
}