#include "ObjectEditor.h"

#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Core\ClassDB.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Misc\ImGuiHelper.h"
#include "Editor\Panels\Scene\SceneArea.h"
#include "Editor\Serialization\SerializedObject.h"
#include "Editor\Serialization\SerializedProperty.h"
#include "Editor\Inspector\ObjectEditorDB.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	Dictionary<ObjectId, ObjectEditor*> ObjectEditor::s_Editors = {};
	Dictionary<size_t, ObjectEditor*> ObjectEditor::s_DefaultEditors = {};

	Texture* ObjectEditor::GetIcon(Object* object)
	{
		return nullptr;
	}

	void ObjectEditor::DrawScene(Object* object)
	{
		m_Object = object;
		OnDrawScene();
	}

	void ObjectEditor::Enable()
	{
		OnEnable();
	}

	void ObjectEditor::DrawInspector()
	{
		OnDrawInspector();
	}

	void ObjectEditor::DrawSceneSelected()
	{
		OnDrawSceneSelected();
	}

	void ObjectEditor::OnPrepareTargets(const List<Object*>& targets)
	{
	}

	void ObjectEditor::OnEnable()
	{
	}

	void ObjectEditor::OnDisable()
	{
	}

	void ObjectEditor::OnDrawInspector()
	{
		SerializedProperty iterator = m_SerializedObject->GetIterator();
		while (iterator.Next())
		{
			ImGui::BeginChangeCheck();
			ImGui::Property(&iterator);
			if (ImGui::EndChangeCheck())
			{
				m_SerializedObject->ApplyModifiedProperties();
				AssetDB::SetDirty(m_SerializedObject->GetTarget());
				SceneArea::RequestRedrawAll();
			}
		} 
	}

	void ObjectEditor::OnDrawScene()
	{
	}

	void ObjectEditor::OnDrawSceneSelected()
	{
	}

	ObjectEditor* ObjectEditor::GetEditor(Object* object)
	{
		ObjectId objectId = object->GetObjectId();
		auto it = s_Editors.find(objectId);
		if (it != s_Editors.end())
		{
			return it->second;
		}
		size_t type = object->GetType();
		const ObjectEditorInfo* info = ObjectEditorDB::GetInfo(type);
		if (info != nullptr)
		{
			ObjectEditor* editor = info->createInstance();
			editor->OnPrepareTargets(List<Object*> { object });
			editor->m_SerializedObject = std::make_shared<SerializedObject>(object);
			editor->OnEnable();
			s_Editors.insert_or_assign(objectId, editor);
			return editor;
		}
		return nullptr;
	}

	ObjectEditor* ObjectEditor::GetEditor(const List<Object*>& objects)
	{
		if (objects.size() == 0)
		{
			return nullptr;
		}
		ObjectId objectId = objects[0]->GetObjectId();
		auto it = s_Editors.find(objectId);
		if (it != s_Editors.end())
		{
			return it->second;
		}
		size_t type = objects[0]->GetType();
		const ObjectEditorInfo* info = ObjectEditorDB::GetInfo(type);
		if (info != nullptr)
		{
			ObjectEditor* editor = info->createInstance();
			editor->OnPrepareTargets(objects);
			editor->m_SerializedObject = std::make_shared<SerializedObject>(objects);
			editor->OnEnable();
			s_Editors.insert_or_assign(objectId, editor);
			return editor;
		}
		return nullptr;
	}

	ObjectEditor* ObjectEditor::GetDefaultEditor(Object* object)
	{
		size_t type = object->GetType();
		auto it = s_DefaultEditors.find(type);
		if (it != s_DefaultEditors.end())
		{
			return it->second;
		}
		const ObjectEditorInfo* info = ObjectEditorDB::GetInfo(type);
		if (info != nullptr)
		{
			ObjectEditor* editor = info->createInstance();
			s_DefaultEditors.insert_or_assign(type, editor);
			return editor;
		}
		return nullptr;
	}

	void ObjectEditor::ReleaseEditor(ObjectEditor* editor)
	{
		ObjectId objectId = editor->m_SerializedObject->GetTarget()->GetObjectId();
		auto it = s_Editors.find(objectId);
		if (it != s_Editors.end())
		{
			editor->OnDisable();
			delete it->second;
			s_Editors.erase(it);
		}
	}
}