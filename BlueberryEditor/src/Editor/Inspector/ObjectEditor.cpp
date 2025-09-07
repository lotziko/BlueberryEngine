#include "ObjectEditor.h"

#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Core\Variant.h"
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
		if (m_HasPadding)
		{
			ImGui::Dummy(ImVec2(3, 0));
			ImGui::SameLine();
			ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0, 0, 0, 0));
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(3, 6));
			ImGui::BeginChild("Inspector", ImVec2(0, 0), ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY);
			ImGui::PopStyleColor();
		}
		OnDrawInspector();
		if (m_HasPadding)
		{
			ImGui::EndChild();
			ImGui::PopStyleVar();
		}
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
		const ObjectEditorInfo& info = ObjectEditorDB::GetInfo(type);
		if (info.createInstance != nullptr)
		{
			ObjectEditor* editor = info.createInstance();
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
		const ObjectEditorInfo& info = ObjectEditorDB::GetInfo(type);
		if (info.createInstance != nullptr)
		{
			ObjectEditor* editor = info.createInstance();
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
		const ObjectEditorInfo& info = ObjectEditorDB::GetInfo(type);
		if (info.createInstance != nullptr)
		{
			ObjectEditor* editor = info.createInstance();
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

	void ObjectEditor::DrawField(Object* object, FieldInfo& info)
	{
		/*String name = info.name;
		if (name.rfind("m_", 0) == 0)
		{
			name.replace(0, 2, "");
		}
		const char* nameLabel = name.c_str();

		Variant value = Variant(object, info.offset);
		bool hasChanged = false;
		switch (info.type)
		{
		case BindingType::Bool:
			if (ImGui::BoolEdit(nameLabel, value.Get<bool>()))
			{
				hasChanged = true;
			}
			break;
		case BindingType::Int:
			if (ImGui::IntEdit(nameLabel, value.Get<int>()))
			{
				hasChanged = true;
			}
			break;
		case BindingType::Float:
			if (ImGui::FloatEdit(nameLabel, value.Get<float>()))
			{
				hasChanged = true;
			}
			break;
		case BindingType::Enum:
			if (ImGui::EnumEdit(nameLabel, value.Get<int>(), static_cast<List<String>*>(info.options.hintData)))
			{
				hasChanged = true;
			}
			break;
		case BindingType::Vector3:
			if (ImGui::DragVector3(nameLabel, value.Get<Vector3>()))
			{
				hasChanged = true;
			}
			break;
		case BindingType::Color:
			if (ImGui::ColorEdit(nameLabel, value.Get<Color>()))
			{
				hasChanged = true;
			}
			break;
		case BindingType::ObjectPtr:
			if (ImGui::ObjectEdit(nameLabel, value.Get<ObjectPtr<Object>>(), info.options.objectType))
			{
				hasChanged = true;
			}
		break;
		case BindingType::ObjectPtrList:
			if (ImGui::ObjectArrayEdit(nameLabel, value.Get<List<ObjectPtr<Object>>>(), info.options.objectType))
			{
				hasChanged = true;
			}
			break;
		}
		if (hasChanged)
		{
			AssetDB::SetDirty(object);
			SceneArea::RequestRedrawAll();
		}*/
	}
}