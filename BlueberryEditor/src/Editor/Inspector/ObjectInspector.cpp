#include "ObjectInspector.h"

#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Core\Variant.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Misc\ImGuiHelper.h"
#include "Editor\Panels\Scene\SceneArea.h"
#include "imgui\imgui.h"

namespace Blueberry
{
	Texture* ObjectInspector::GetIcon(Object* object)
	{
		return nullptr;
	}

	void ObjectInspector::Draw(Object* object)
	{
		auto info = ClassDB::GetInfo(object->GetType());
		for (auto& field : info.fields)
		{
			if (!field.options.isHidden)
			{
				DrawField(object, field);
			}
		}
	}

	void ObjectInspector::DrawScene(Object* object)
	{
	}

	void ObjectInspector::DrawField(Object* object, FieldInfo& info)
	{
		String name = info.name;
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
		case BindingType::ObjectPtrArray:
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
		}
	}
}