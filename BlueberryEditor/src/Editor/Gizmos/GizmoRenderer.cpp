#include "GizmoRenderer.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Component.h"
#include "Blueberry\Graphics\Buffers\PerCameraDataConstantBuffer.h"

#include "Editor\Selection.h"
#include "Editor\Gizmos\Gizmos.h"
#include "Editor\Inspector\ObjectEditorDB.h"
#include "Editor\Inspector\ObjectEditor.h"

#include "Blueberry\Graphics\RendererTree.h"

#include "Blueberry\Scene\Scene.h"

namespace Blueberry
{
	void GizmoRenderer::Draw(Scene* scene, Camera* camera, GfxTexture* target)
	{
		// TODO add cache and draw also child entities
		PerCameraDataConstantBuffer::BindData(camera, target);
		Object* activeObject = Selection::GetActiveObject();
		if (activeObject != nullptr)
		{
			if (activeObject->IsClassType(Entity::Type))
			{
				Entity* entity = static_cast<Entity*>(activeObject);
				if (entity->IsActiveInHierarchy())
				{
					for (uint32_t i = 0; i < entity->GetComponentCount(); ++i)
					{
						ObjectEditor* editor = ObjectEditor::GetEditor(entity->GetComponentAt(i));
						if (editor != nullptr)
						{
							Gizmos::SetColor(Color(1, 1, 1, 1));
							Gizmos::Begin();
							editor->DrawSceneSelected();
							Gizmos::End();
						}
					}
				}
			}
		}
	}
}