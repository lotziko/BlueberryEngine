#include "bbpch.h"
#include "GizmoRenderer.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Component.h"
#include "Blueberry\Graphics\PerCameraDataConstantBuffer.h"

#include "Editor\Selection.h"
#include "Editor\Gizmos\Gizmos.h"
#include "Editor\Inspector\ObjectInspectorDB.h"
#include "Editor\Inspector\ObjectInspector.h"

#include "Blueberry\Graphics\RendererTree.h"

namespace Blueberry
{
	void GizmoRenderer::Draw(Scene* scene, Camera* camera)
	{
		PerCameraDataConstantBuffer::BindData(camera);

		// TODO add cache and draw also child entities

		/*Gizmos::SetMatrix(Matrix::Identity);
		Gizmos::SetColor(Color(1, 1, 1, 1));
		Gizmos::Begin();
		std::vector<AABB> bounds = {};
		RendererTree::GatherBounds(bounds);
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