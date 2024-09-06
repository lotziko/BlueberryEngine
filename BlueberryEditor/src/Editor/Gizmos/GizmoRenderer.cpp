#include "bbpch.h"
#include "GizmoRenderer.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Component.h"
#include "Blueberry\Graphics\PerCameraDataConstantBuffer.h"

#include "Editor\Selection.h"
#include "Editor\Gizmos\Gizmos.h"
#include "Editor\Inspector\ObjectInspectorDB.h"
#include "Editor\Inspector\ObjectInspector.h"

namespace Blueberry
{
	void GizmoRenderer::Draw(Scene* scene, BaseCamera* camera)
	{
		PerCameraDataConstantBuffer::BindData(camera);

		Object* activeObject = Selection::GetActiveObject();
		if (activeObject != nullptr)
		{
			if (activeObject->IsClassType(Entity::Type))
			{
				Entity* entity = static_cast<Entity*>(activeObject);
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