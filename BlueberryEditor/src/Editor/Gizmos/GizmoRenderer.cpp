#include "GizmoRenderer.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\Component.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Graphics\Buffers\PerCameraDataConstantBuffer.h"

#include "Editor\Selection.h"
#include "Editor\Gizmos\Gizmos.h"
#include "Editor\Inspector\ObjectEditorDB.h"
#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	static List<ObjectEditor*> s_ActiveEditors;

	void GizmoRenderer::GatherAndUpdate(Scene* scene)
	{
		s_ActiveEditors.clear();
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
						Component* component = entity->GetComponentAt(i);
						ObjectEditor* editor = ObjectEditor::GetEditor(component);
						if (editor != nullptr)
						{
							s_ActiveEditors.push_back(editor);
						}
					}
				}
			}
		}

		if (!s_ActiveEditors.empty())
		{
			Gizmos::Begin(GizmoEvent::Search);
			for (ObjectEditor* editor : s_ActiveEditors)
			{
				editor->DrawSceneSelected();
			}
			Gizmos::End();
			Gizmos::Begin(GizmoEvent::Input);
			for (ObjectEditor* editor : s_ActiveEditors)
			{
				editor->DrawSceneSelected();
			}
			Gizmos::End();
		}
	}
	
	void GizmoRenderer::Draw(GfxTexture* target)
	{
		Camera* camera = Camera::GetCurrent();
		PerCameraDataConstantBuffer::BindData(camera, target);
		if (!s_ActiveEditors.empty())
		{
			Gizmos::Begin(GizmoEvent::Draw);
			for (ObjectEditor* editor : s_ActiveEditors)
			{
				editor->DrawSceneSelected();
			}
			Gizmos::End();
		}
	}
}