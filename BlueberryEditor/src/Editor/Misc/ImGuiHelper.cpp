#include "ImGuiHelper.h"

#include "Blueberry\Core\Screen.h"
#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Core\ClassDB.h"

#include "Editor\Panels\Picking\ObjectPicker.h"
#include "Editor\Panels\Project\ProjectBrowser.h"
#include "Editor\Serialization\SerializedProperty.h"
#include "Editor\Assets\AssetDB.h"

#include <imgui\imgui_internal.h>
#include <imgui\misc\freetype\imgui_freetype.h>
#include <imgui\misc\cpp\imgui_stdlib.h>

ImGui::EditorContext* ImGui::GEditor = NULL;
ImGui::ClearOverrideEvent ImGui::Events::s_ClearedOverride = {};

static Blueberry::List<bool> s_ChangeStack = {};
static bool s_MixedValue = false;
static bool s_ShowPopup = false;
static const bool* s_MixedValueMask = {};
static ImVector<ImRect> s_PaddingStack;

#define PROPERTY_LABEL()\
ImGui::PushID(label);\
float availableWidth = ImGui::GetContentRegionAvail().x;\
float labelWidth = std::max(150.0f, availableWidth * 0.4f);\
float valueWidth = std::max(0.0f, availableWidth - labelWidth);\
ImGui::Text(label);\
if (ImGui::IsItemClicked(ImGuiMouseButton_Right))\
{\
	s_ShowPopup = true;\
}\
ImGui::SameLine(labelWidth);\

#define PROPERTY_BEGIN_VALUE()\
if (s_MixedValue)\
{\
	ImGui::PushItemFlag(ImGuiItemFlags_MixedValue, true);\
}\
ImGui::SetNextItemWidth(valueWidth);\

#define PROPERTY_END_VALUE()\
if (s_MixedValue)\
{\
	ImGui::PopItemFlag();\
}\
ImGui::PopID();\

Blueberry::SerializedProperty* ImGui::ClearOverrideEventArgs::GetProperty()
{
	return m_Property;
}

ImGui::ClearOverrideEvent& ImGui::Events::GetClearedOverride()
{
	return s_ClearedOverride;
}

void ImGui::CreateEditorContext()
{
	GEditor = static_cast<EditorContext*>(BB_MALLOC(sizeof(ImGui::EditorContext)));
	s_PaddingStack.push_back(ImRect());
}

ImGui::EditorStyle& ImGui::GetEditorStyle()
{
	return GEditor->Style;
}

bool ImGui::Property(Blueberry::SerializedProperty* property)
{
	Blueberry::String name = property->GetName();
	if (name.rfind("m_", 0) == 0)
	{
		name.replace(0, 2, "");
	}
	return Property(property, name.c_str());
}

ImVec2 GetPropertyHeight(Blueberry::SerializedProperty* property)
{
	if (property->GetType() == Blueberry::BindingType::DataList)
	{
		return ImVec2(0, 0);
	}
	return ImVec2(ImGui::GetContentRegionAvail().x, property->GetListSize() * (ImGui::GetFontSize() + ImGui::GetStyle().FramePadding.y * 2.0f + ImGui::GetStyle().ItemSpacing.y) + 6);
}

bool ImGui::Property(Blueberry::SerializedProperty* property, const char* label)
{
	ImGui::PushID(static_cast<int>(property->GetId()));
	if (property->IsOverriden())
	{
		ImVec2 screenPos = ImGui::GetCursorScreenPos() - ImVec2(s_PaddingStack.back().Min.x, 0);
		ImVec2 min = screenPos;
		ImVec2 max = screenPos + ImVec2(4, ImGui::GetTextLineHeightWithSpacing());
		ImGui::GetWindowDrawList()->AddRectFilled(min, max, ImGui::GetColorU32(ImGuiCol_SeparatorActive));
	}
	s_MixedValue = property->IsMixedValue();
	s_MixedValueMask = property->GetMixedMask();
	bool result = false;
	switch (property->GetType())
	{
	case Blueberry::BindingType::Bool:
	{
		bool value = property->GetBool();
		if (ImGui::BoolEdit(label, &value))
		{
			property->SetBool(value);
			result = true;
		}
	}
	break;
	case Blueberry::BindingType::Int:
	{
		int value = property->GetInt();
		if (ImGui::IntEdit(label, &value))
		{
			property->SetInt(value);
			result = true;
		}
	}
	break;
	case Blueberry::BindingType::Uint:
	{
		uint32_t value = property->GetUint();
		if (ImGui::UintEdit(label, &value))
		{
			property->SetUint(value);
			result = true;
		}
	}
	break;
	case Blueberry::BindingType::Float:
	{
		float value = property->GetFloat();
		if (ImGui::FloatEdit(label, &value))
		{
			property->SetFloat(value);
			result = true;
		}
	}
	break;
	case Blueberry::BindingType::Enum:
	{
		int value = property->GetInt();
		if (ImGui::EnumEdit(label, &value, static_cast<Blueberry::List<Blueberry::String>*>(property->GetHintData())))
		{
			property->SetInt(value);
			result = true;
		}
	}
	break;
	case Blueberry::BindingType::String:
	{
		Blueberry::String value = property->GetString();
		if (ImGui::StringEdit(label, &value)) // TODO mixed
		{
			property->SetString(Blueberry::String(value.data()));
			result = true;
		}
	}
	break;
	case Blueberry::BindingType::Vector2:
	{
		Blueberry::Vector2 value = property->GetVector2();
		if (ImGui::DragVector2(label, &value))
		{
			property->SetVector2(value);
			result = true;
		}
	}
	break;
	case Blueberry::BindingType::Vector3:
	{
		Blueberry::Vector3 value = property->GetVector3();
		if (ImGui::DragVector3(label, &value))
		{
			property->SetVector3(value);
			result = true;
		}
	}
	break;
	case Blueberry::BindingType::Vector4:
	{
		Blueberry::Vector4 value = property->GetVector4();
		if (ImGui::DragVector4(label, &value))
		{
			property->SetVector4(value);
			result = true;
		}
	}
	break;
	case Blueberry::BindingType::Quaternion:
	{
		Blueberry::Quaternion value = property->GetQuaternion();
		if (ImGui::DragVector4(label, &static_cast<Blueberry::Vector4>(value)))
		{
			property->SetQuaternion(value);
			result = true;
		}
	}
	break;
	case Blueberry::BindingType::Color:
	{
		Blueberry::Color value = property->GetColor();
		if (ImGui::ColorEdit(label, &value))
		{
			property->SetColor(value);
			result = true;
		}
	}
	break;
	case Blueberry::BindingType::ObjectPtr:
	{
		Blueberry::ObjectPtr<Blueberry::Object> value = property->GetObjectPtr();
		if (ImGui::ObjectEdit(label, &value, property->GetObjectPtrType()))
		{
			property->SetObjectPtr(value);
			result = true;
		}
	}
	break;
	case Blueberry::BindingType::Data:
	{
		ImGui::Text(label);
		Blueberry::SerializedProperty childProperty = *property;
		size_t depth = childProperty.GetDepth() + 1;
		while (childProperty.Next(true))
		{
			if (childProperty.GetDepth() <= depth)
			{
				break;
			}
			ImGui::Property(&childProperty);
		}
	}
	break;
	case Blueberry::BindingType::FloatList:
	case Blueberry::BindingType::StringList:
	case Blueberry::BindingType::Vector2List:
	case Blueberry::BindingType::Vector3List:
	case Blueberry::BindingType::Vector4List:
	case Blueberry::BindingType::ObjectPtrList:
	case Blueberry::BindingType::DataList:
	{
		ImGui::Text(label);
		ImGui::BeginChild("##list", GetPropertyHeight(property), ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY);
		ImGui::BeginPaddedArea(ImVec2(10, 5), ImVec2(4, 1));
		Blueberry::SerializedProperty listProperty = *property;
		for (size_t i = 0; i < listProperty.GetListSize(); ++i)
		{
			ImGui::Property(&listProperty.GetListElement(i));
			property->Next(false);
		}
		ImGui::EndPaddedArea();
		ImGui::EndChild();
		if (ImGui::Button("+"))
		{
			listProperty.InsertListElement(listProperty.GetListSize());
			TriggerChange();
		}
		ImGui::SameLine();
		if (ImGui::Button("-"))
		{
			listProperty.DeleteListElement(listProperty.GetListSize() - 1);
			TriggerChange();
		}
	}
	break;
	default:
		ImGui::Text(label);
		break;
	}
	const char* popupId = "PropertyPopup";
	if (s_ShowPopup)
	{
		ImGui::OpenPopup(popupId);
		// gather items
		s_ShowPopup = false;
	}
	if (ImGui::BeginPopup(popupId))
	{
		bool hasAnyItem = false;
		if (property->IsOverriden())
		{
			if (ImGui::MenuItem("Clear override"))
			{
				property->ClearOverride();
				auto& event = ImGui::Events::GetClearedOverride();
				if (event.HasCallbacks())
				{
					ClearOverrideEventArgs args(property);
					event.Invoke(args);
				}
				TriggerChange();
			}
			hasAnyItem = true;
		}
		if (!hasAnyItem)
		{
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}
	s_MixedValue = false;
	ImGui::PopID();
	return result;
}

void ImGui::BeginChangeCheck()
{
	s_ChangeStack.push_back(false);
}

void ImGui::TriggerChange()
{
	for (uint32_t i = 0; i < s_ChangeStack.size(); ++i)
	{
		s_ChangeStack[i] = true;
	}
}

bool ImGui::EndChangeCheck()
{
	if (s_ChangeStack.size() > 0)
	{
		bool value = s_ChangeStack[s_ChangeStack.size() - 1];
		s_ChangeStack.pop_back();
		return value;
	}
	return false;
}

void ImGui::SetMixedValue(const bool& mixed)
{
	s_MixedValue = mixed;
}

bool ImGui::BeginPopup(ImGuiID id, ImGuiWindowFlags flags)
{
	// Copy from imgui.cpp
	ImGuiContext& g = *GImGui;
	if (g.OpenPopupStack.Size <= g.BeginPopupStack.Size) // Early out for performance
	{
		g.NextWindowData.ClearFlags(); // We behave like Begin() and need to consume those values
		return false;
	}
	flags |= ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings;
	return BeginPopupEx(id, flags);
}

void ImGui::BeginPaddedArea(ImVec2 min, ImVec2 max)
{
	ImGui::SetCursorPos(ImGui::GetCursorPos() + min);
	s_PaddingStack.push_back(ImRect(min, max));
	ImGui::GetCurrentWindow()->ContentRegionRect.Max -= ImVec2(max.x, 0);
	ImGui::BeginGroup();
}

void ImGui::EndPaddedArea()
{
	ImGui::EndGroup();
	ImGui::SetCursorPosY(ImGui::GetCursorPosY() + s_PaddingStack.back().Max.y);
	ImGui::GetCurrentWindow()->ContentRegionRect.Max += ImVec2(s_PaddingStack.back().Max.x, 0);
	s_PaddingStack.pop_back();
}

bool ImGui::DragVector2(const char* label, Blueberry::Vector2* v)
{
	PROPERTY_LABEL()
	PROPERTY_BEGIN_VALUE()

	bool result = false;
	float vector2[2] = { v->x, v->y };
	if (ImGui::DragVectorN("##vector2", ImGuiDataType_Float, 2, vector2))
	{
		v->x = vector2[0];
		v->y = vector2[1];
		result = true;
	}

	PROPERTY_END_VALUE()
	if (result)
	{
		TriggerChange();
	}
	return result;
}

bool ImGui::DragVector3(const char* label, Blueberry::Vector3* v)
{
	PROPERTY_LABEL()
	PROPERTY_BEGIN_VALUE()

	bool result = false;
	float vector3[3] = { v->x, v->y, v->z };

	if (ImGui::DragVectorN("##vector3", ImGuiDataType_Float, 3, vector3))
	{
		v->x = vector3[0];
		v->y = vector3[1];
		v->z = vector3[2];
		result = true;
	}

	PROPERTY_END_VALUE()
	if (result)
	{
		TriggerChange();
	}
	return result;
}

bool ImGui::DragVector4(const char* label, Blueberry::Vector4* v)
{
	PROPERTY_LABEL()
	PROPERTY_BEGIN_VALUE()

	bool result = false;
	float vector4[4] = { v->x, v->y, v->z, v->w };

	if (ImGui::DragVectorN("##vector3", ImGuiDataType_Float, 4, vector4))
	{
		v->x = vector4[0];
		v->y = vector4[1];
		v->z = vector4[2];
		v->w = vector4[3];
		result = true;
	}

	PROPERTY_END_VALUE()
	if (result)
	{
		TriggerChange();
	}
	return result;
}

bool ImGui::DragVectorN(const char* label, ImGuiDataType dataType, int components, void* data)
{
	ImGuiWindow* window = ImGui::GetCurrentWindow();
	if (window->SkipItems)
		return false;

	ImGuiContext& g = *GImGui;
	bool result = false;
	ImGui::BeginGroup();
	ImGui::PushID(label);
	ImGui::PushMultiItemsWidths(components, ImGui::CalcItemWidth());// - s_PaddingStack.back().Max.x);
	size_t typeSize = ImGui::DataTypeGetInfo(dataType)->Size;
	for (int i = 0; i < components; i++)
	{
		if (i > 0)
		{
			ImGui::SameLine(0, g.Style.ItemInnerSpacing.x);
		}
		ImGui::PushID(i);
		ImGui::PushItemFlag(ImGuiItemFlags_MixedValue, s_MixedValueMask[i]);
		if (ImGui::DragScalar("", dataType, data, 0.1f, 0, 0, "%.3f"))
		{
			result = true;
		}
		data = (void*)((char*)data + typeSize);
		ImGui::PopItemFlag();
		ImGui::PopID();
		ImGui::PopItemWidth();
	}
	ImGui::PopID();
	ImGui::EndGroup();
	return result;
}

bool ImGui::InputEnum(const char* label, int* v, const Blueberry::List<Blueberry::String>* names)
{
	bool result = false;
	if (ImGui::BeginCombo("##enum", names->at(*v).c_str()))
	{
		for (int i = 0; i < names->size(); i++)
		{
			bool isSelected = *v == i;
			if (ImGui::Selectable(names->at(i).c_str(), isSelected))
			{
				*v = i;
				result = true;
			}

			if (isSelected)
			{
				ImGui::SetItemDefaultFocus();
			}
		}
		ImGui::EndCombo();
	}
	return result;
}

bool ImGui::InputEnum(const char* label, int* v, const Blueberry::List<std::pair<Blueberry::String, int>>* nameValues)
{
	int value = *v;
	int size = static_cast<int>(nameValues->size());
	const char* preview = nullptr;
	for (int i = 0; i < size; i++)
	{
		auto& pair = nameValues->at(i);
		if (pair.second == value)
		{
			preview = nameValues->at(i).first.c_str();
			break;
		}
	}

	bool result = false;
	if (ImGui::BeginCombo("##enum", preview))
	{
		for (int i = 0; i < size; i++)
		{
			auto& pair = nameValues->at(i);
			bool isSelected = value == pair.second;
			if (ImGui::Selectable(pair.first.c_str(), isSelected))
			{
				*v = pair.second;
				result = true;
			}

			if (isSelected)
			{
				ImGui::SetItemDefaultFocus();
			}
		}
		ImGui::EndCombo();
	}
	return result;
}

bool ImGui::EnumEdit(const char* label, int* v, const Blueberry::List<Blueberry::String>* names)
{
	if (names == nullptr || names->size() == 0)
	{
		return false;
	}

	PROPERTY_LABEL()
	PROPERTY_BEGIN_VALUE()

	bool result = InputEnum(label, v, names);

	PROPERTY_END_VALUE()
	if (result)
	{
		TriggerChange();
	}
	return result;
}

bool ImGui::EnumEdit(const char* label, int* v, const Blueberry::List<std::pair<Blueberry::String, int>>* nameValues)
{
	if (nameValues == nullptr || nameValues->size() == 0)
	{
		return false;
	}

	PROPERTY_LABEL()
	PROPERTY_BEGIN_VALUE()

	bool result = InputEnum(label, v, nameValues);

	PROPERTY_END_VALUE()
	if (result)
	{
		TriggerChange();
	}
	return result;
}

bool ImGui::BoolEdit(const char* label, bool* v)
{
	PROPERTY_LABEL()
	PROPERTY_BEGIN_VALUE()

	bool result = false;
	if (ImGui::Checkbox("##bool", v))
	{
		result = true;
	}

	PROPERTY_END_VALUE()
	if (result)
	{
		TriggerChange();
	}
	return result;
}

bool ImGui::IntEdit(const char* label, int* v)
{
	PROPERTY_LABEL()
	PROPERTY_BEGIN_VALUE()

	bool result = false;
	if (ImGui::DragInt("##int", v))
	{
		result = true;
	}

	PROPERTY_END_VALUE()
	if (result)
	{
		TriggerChange();
	}
	return result;
}

bool ImGui::UintEdit(const char* label, uint32_t* v)
{
	PROPERTY_LABEL()
	PROPERTY_BEGIN_VALUE()

	bool result = false;
	if (ImGui::DragScalar("##uint", ImGuiDataType_U32, v))
	{
		result = true;
	}

	PROPERTY_END_VALUE()
	if (result)
	{
		TriggerChange();
	}
	return result;
}

bool ImGui::FloatEdit(const char* label, float* v, float min, float max)
{
	PROPERTY_LABEL()
	PROPERTY_BEGIN_VALUE()
	
	bool result = false;
	if (ImGui::DragFloat("##float", v, 0.02f, min, max))
	{
		result = true;
	}

	PROPERTY_END_VALUE()
	if (result)
	{
		TriggerChange();
	}
	return result;
}

bool ImGui::ColorEdit(const char* label, Blueberry::Color* v)
{
	PROPERTY_LABEL()
	PROPERTY_BEGIN_VALUE()

	bool result = false;
	float color4[4] = { v->R(), v->G(), v->B(), v->A() };
	if (ImGui::ColorEdit4("##color4", color4))
	{
		v->R(color4[0]);
		v->G(color4[1]);
		v->B(color4[2]);
		v->A(color4[3]);
		result = true;
	}

	PROPERTY_END_VALUE()
	if (result)
	{
		TriggerChange();
	}
	return result;
}

bool ImGui::StringEdit(const char* label, Blueberry::String* v)
{
	PROPERTY_LABEL()
	PROPERTY_BEGIN_VALUE()
	
	bool result = false;
	if (ImGui::InputText("##string", v))
	{
		result = true;
	}

	PROPERTY_END_VALUE()
	if (result)
	{
		TriggerChange();
	}
	return result;
}

bool ImGui::ObjectEdit(const char* label, Blueberry::Object** v, const Blueberry::TypeId& type)
{
	PROPERTY_LABEL()
	PROPERTY_BEGIN_VALUE()

	bool result = false;
	Blueberry::Object* vObj = *v;
	ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetColorU32(ImGuiCol_FrameBg));
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetColorU32(ImGuiCol_FrameBgHovered));
	ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));
	if (ImGui::Button(vObj == nullptr ? "None" : vObj->GetName().c_str(), ImVec2(ImGui::CalcItemWidth(), 0)))
	{
		Blueberry::ObjectPicker::Open(v, type);
	}
	ImGui::PopStyleVar();
	ImGui::PopStyleColor(2);
	if (Blueberry::ObjectPicker::GetResult(v))
	{
		vObj = *v;
		if (vObj != nullptr && vObj->GetState() == Blueberry::ObjectState::AwaitingLoading && Blueberry::ObjectDB::HasGuid(vObj))
		{
			Blueberry::AssetDB::LoadAsset(Blueberry::ObjectDB::GetGuidFromObject(vObj));
		}
		result = true;
	}
	
	if (ImGui::BeginPopupContextItem())
	{
		if (ImGui::MenuItem("Show in project"))
		{
			Blueberry::ProjectBrowser::ShowObject(vObj);
		}
		if (!Blueberry::ObjectDB::HasGuid(vObj))
		{
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}

	if (ImGui::BeginDragDropTarget())
	{
		const ImGuiPayload* payload = ImGui::GetDragDropPayload();
		if (payload != nullptr && payload->IsDataType("OBJECT_ID"))
		{
			size_t count = payload->DataSize / sizeof(Blueberry::ObjectId);
			Blueberry::ObjectId* ids = static_cast<Blueberry::ObjectId*>(payload->Data);
			for (size_t i = 0; i < count; ++i)
			{
				Blueberry::Object* object = Blueberry::ObjectDB::GetObject(ids[i]);
				if (object != nullptr && object->IsClassType(type) && ImGui::AcceptDragDropPayload("OBJECT_ID"))
				{
					if (object->GetState() == Blueberry::ObjectState::AwaitingLoading && Blueberry::ObjectDB::HasGuid(object))
					{
						Blueberry::AssetDB::LoadAsset(Blueberry::ObjectDB::GetGuidFromObject(object));
					}

					*v = object;
					result = true;
					break;
				}
			}
		}
		ImGui::EndDragDropTarget();
	}

	PROPERTY_END_VALUE()
	if (result)
	{
		TriggerChange();
	}
	return result;
}

bool ImGui::ObjectEdit(const char* label, Blueberry::ObjectPtr<Blueberry::Object>* v, const Blueberry::TypeId& type)
{
	Blueberry::Object* object = v->Get();
	if (ObjectEdit(label, &object, type))
	{
		*v = object;
		return true;
	}
	return false;
}

bool ImGui::ObjectArrayEdit(const char* label, Blueberry::List<Blueberry::ObjectPtr<Blueberry::Object>>* v, const Blueberry::TypeId& type)
{
	ImGui::Text(label);
	for (int i = 0; i < v->size(); ++i)
	{
		Blueberry::ObjectPtr<Blueberry::Object>* objectPtr = (v->data() + i);
		if (ObjectEdit("", objectPtr, type))
		{
			return true;
		}
	}
	return false;
}

struct InputTextCallback_UserData
{
	Blueberry::String* Str;
	ImGuiInputTextCallback  ChainCallback;
	void* ChainCallbackUserData;
};

static int InputTextCallback(ImGuiInputTextCallbackData* data)
{
	InputTextCallback_UserData* user_data = (InputTextCallback_UserData*)data->UserData;
	if (data->EventFlag == ImGuiInputTextFlags_CallbackResize)
	{
		// Resize string callback
		// If for some reason we refuse the new length (BufTextLen) and/or capacity (BufSize) we need to set them back to what we want.
		Blueberry::String* str = user_data->Str;
		IM_ASSERT(data->Buf == str->c_str());
		str->resize(data->BufTextLen);
		data->Buf = (char*)str->c_str();
	}
	else if (user_data->ChainCallback)
	{
		// Forward to user callback, if any
		data->UserData = user_data->ChainCallbackUserData;
		return user_data->ChainCallback(data);
	}
	return 0;
}

bool ImGui::InputText(const char* label, Blueberry::String* str, ImGuiInputTextFlags flags, ImGuiInputTextCallback callback, void* user_data)
{
	IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
	flags |= ImGuiInputTextFlags_CallbackResize;

	InputTextCallback_UserData cb_user_data;
	cb_user_data.Str = str;
	cb_user_data.ChainCallback = callback;
	cb_user_data.ChainCallbackUserData = user_data;
	return InputText(label, (char*)str->c_str(), str->capacity() + 1, flags, InputTextCallback, &cb_user_data);
}

bool ImGui::InputTextMultiline(const char* label, Blueberry::String* str, const ImVec2& size, ImGuiInputTextFlags flags, ImGuiInputTextCallback callback, void* user_data)
{
	IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
	flags |= ImGuiInputTextFlags_CallbackResize;

	InputTextCallback_UserData cb_user_data;
	cb_user_data.Str = str;
	cb_user_data.ChainCallback = callback;
	cb_user_data.ChainCallbackUserData = user_data;
	return InputTextMultiline(label, (char*)str->c_str(), str->capacity() + 1, size, flags, InputTextCallback, &cb_user_data);
}

bool ImGui::InputTextWithHint(const char* label, const char* hint, Blueberry::String* str, ImGuiInputTextFlags flags, ImGuiInputTextCallback callback, void* user_data)
{
	IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
	flags |= ImGuiInputTextFlags_CallbackResize;

	InputTextCallback_UserData cb_user_data;
	cb_user_data.Str = str;
	cb_user_data.ChainCallback = callback;
	cb_user_data.ChainCallbackUserData = user_data;
	return InputTextWithHint(label, hint, (char*)str->c_str(), str->capacity() + 1, flags, InputTextCallback, &cb_user_data);
}

bool ImGui::SearchInputText(const char* hint, Blueberry::String* text)
{
	// https://github.com/ocornut/imgui/issues/7510
	ImGuiID id = ImGui::GetID("###search");
	bool hovered = GImGui->HoveredIdPreviousFrame == id;
	bool focused = GImGui->ActiveId == id;
	if (focused)
	{
		ImGui::PushStyleColor(ImGuiCol_Border, ImGui::GetStyleColorVec4(ImGuiCol_SeparatorActive));
	}
	else if (!hovered)
	{
		ImVec4 color = ImGui::GetStyleColorVec4(ImGuiCol_FrameBg);
		color.x *= 0.9f;
		color.y *= 0.9f;
		color.z *= 0.9f;
		ImGui::PushStyleColor(ImGuiCol_FrameBg, color);
	}
	bool result = ImGui::InputTextWithHint("###search", hint, text);
	if (focused)
	{
		ImGui::PopStyleColor();
	}
	else if (!hovered)
	{
		ImGui::PopStyleColor();
	}
	if (result)
	{
		TriggerChange();
	}
	return result;
}

void ImGui::HorizontalSplitter(const char* strId, float* size, float minSize)
{
	ImVec2 screenPos = ImGui::GetCursorScreenPos();
	ImVec2 min = ImVec2(screenPos.x, screenPos.y);
	ImVec2 max = ImVec2(screenPos.x, screenPos.y + ImGui::GetContentRegionAvail().y);
	bool hovered, held;
	ImGuiID id = ImGui::GetID(strId);
	ImGui::KeepAliveID(id);

	ImGui::Dummy(ImVec2(1, 1));
	ImGui::ButtonBehavior(ImRect(ImVec2(min.x - 4, min.y), ImVec2(max.x + 4, max.y)), id, &hovered, &held);
	ImGui::GetWindowDrawList()->AddLine(min, max, ImGui::GetColorU32(held ? ImGuiCol_SeparatorActive : (hovered ? ImGuiCol_SeparatorHovered : ImGuiCol_Separator)));
	if (hovered || held)
	{
		ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
	}
	if (held)
	{
		*size = std::max(GImGui->IO.MousePos.x - GImGui->ActiveIdClickOffset.x - 4, minSize);
	}
}

bool ImGui::CenteredButton(const char* label)
{
	// https://github.com/ocornut/imgui/discussions/3862
	ImGuiStyle& style = ImGui::GetStyle();

	float size = ImGui::CalcTextSize(label).x + style.FramePadding.x * 2.0f;
	float avail = ImGui::GetContentRegionAvail().x;

	float off = (avail - size) * 0.5f;
	if (off > 0.0f)
	{
		ImGui::SetCursorPosX(ImGui::GetCursorPosX() + off);
	}

	return ImGui::Button(label);
}

void ImGui::ApplyEditorDarkTheme()
{
	ImGuiStyle* style = &ImGui::GetStyle();
	ImVec4* colors = style->Colors;

	colors[ImGuiCol_Text] = ImVec4(1.000f, 1.000f, 1.000f, 1.000f);
	colors[ImGuiCol_TextDisabled] = ImVec4(0.500f, 0.500f, 0.500f, 1.000f);
	colors[ImGuiCol_WindowBg] = ImVec4(0.180f, 0.180f, 0.180f, 1.000f);
	colors[ImGuiCol_ChildBg] = ImVec4(0.280f, 0.280f, 0.280f, 0.000f);
	colors[ImGuiCol_PopupBg] = ImVec4(0.148f, 0.148f, 0.148f, 1.000f);
	colors[ImGuiCol_Border] = ImVec4(0.266f, 0.266f, 0.266f, 1.000f);
	colors[ImGuiCol_BorderShadow] = ImVec4(0.000f, 0.000f, 0.000f, 0.000f);
	colors[ImGuiCol_FrameBg] = ImVec4(0.160f, 0.160f, 0.160f, 1.000f);
	colors[ImGuiCol_FrameBgHovered] = ImVec4(0.200f, 0.200f, 0.200f, 1.000f);
	colors[ImGuiCol_FrameBgActive] = ImVec4(0.280f, 0.280f, 0.280f, 1.000f);
	colors[ImGuiCol_TitleBg] = ImVec4(0.148f, 0.148f, 0.148f, 1.000f);
	colors[ImGuiCol_TitleBgActive] = ImVec4(0.148f, 0.148f, 0.148f, 1.000f);
	colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.148f, 0.148f, 0.148f, 1.000f);
	colors[ImGuiCol_MenuBarBg] = ImVec4(0.195f, 0.195f, 0.195f, 1.000f);
	colors[ImGuiCol_ScrollbarBg] = ImVec4(0.160f, 0.160f, 0.160f, 1.000f);
	colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.277f, 0.277f, 0.277f, 1.000f);
	colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.300f, 0.300f, 0.300f, 1.000f);
	colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_CheckMark] = ImVec4(1.000f, 1.000f, 1.000f, 1.000f);
	colors[ImGuiCol_SliderGrab] = ImVec4(0.391f, 0.391f, 0.391f, 1.000f);
	colors[ImGuiCol_SliderGrabActive] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_Button] = ImVec4(1.000f, 1.000f, 1.000f, 0.000f);
	colors[ImGuiCol_ButtonHovered] = ImVec4(1.000f, 1.000f, 1.000f, 0.156f);
	colors[ImGuiCol_ButtonActive] = ImVec4(1.000f, 1.000f, 1.000f, 0.391f);
	colors[ImGuiCol_Header] = ImVec4(0.313f, 0.313f, 0.313f, 1.000f);
	colors[ImGuiCol_HeaderHovered] = ImVec4(0.469f, 0.469f, 0.469f, 1.000f);
	colors[ImGuiCol_HeaderActive] = ImVec4(0.469f, 0.469f, 0.469f, 1.000f);
	colors[ImGuiCol_Separator] = colors[ImGuiCol_Border];
	colors[ImGuiCol_SeparatorHovered] = ImVec4(0.391f, 0.391f, 0.391f, 1.000f);
	colors[ImGuiCol_SeparatorActive] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_ResizeGrip] = ImVec4(1.000f, 1.000f, 1.000f, 0.250f);
	colors[ImGuiCol_ResizeGripHovered] = ImVec4(1.000f, 1.000f, 1.000f, 0.670f);
	colors[ImGuiCol_ResizeGripActive] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_Tab] = ImVec4(0.148f, 0.148f, 0.148f, 1.000f);
	colors[ImGuiCol_TabHovered] = ImVec4(0.352f, 0.352f, 0.352f, 1.000f);
	colors[ImGuiCol_TabActive] = ImVec4(0.195f, 0.195f, 0.195f, 1.000f);
	colors[ImGuiCol_TabUnfocused] = ImVec4(0.148f, 0.148f, 0.148f, 1.000f);
	colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.195f, 0.195f, 0.195f, 1.000f);
	colors[ImGuiCol_DockingPreview] = ImVec4(1.000f, 0.391f, 0.000f, 0.781f);
	colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.180f, 0.180f, 0.180f, 1.000f);
	colors[ImGuiCol_PlotLines] = ImVec4(0.469f, 0.469f, 0.469f, 1.000f);
	colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_PlotHistogram] = ImVec4(0.586f, 0.586f, 0.586f, 1.000f);
	colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_TextSelectedBg] = ImVec4(1.000f, 1.000f, 1.000f, 0.156f);
	colors[ImGuiCol_DragDropTarget] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_NavHighlight] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.000f, 0.000f, 0.000f, 0.586f);
	colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.000f, 0.000f, 0.000f, 0.586f);

	// Convert to SRGB
	/*for (int i = 0; i < ImGuiCol_COUNT; i++)
	{
		ImVec4 color = colors[i];
		color.x = powf(color.x, 2.2f);
		color.y = powf(color.y, 2.2f);
		color.z = powf(color.z, 2.2f);
		colors[i] = color;
	}*/

	float scale = Blueberry::Screen::GetScale();

	style->ChildRounding = 4.0f * scale;
	style->FrameBorderSize = 1.0f;
	style->FrameRounding = 2.0f * scale;
	style->GrabMinSize = 7.0f * scale;
	style->PopupRounding = 2.0f * scale;
	style->ScrollbarRounding = 12.0f * scale;
	style->ScrollbarSize = 13.0f * scale;
	style->TabBorderSize = 1.0f * scale;
	style->TabRounding = 0.0f;
	style->WindowRounding = 4.0f * scale;
	style->WindowPadding = ImVec2(0.0f, 0.0f);

	EditorStyle* editorStyle = &GEditor->Style;
	
	editorStyle->ProjectBottomPanelSize = 20 * scale;
	editorStyle->ProjectCellSize = 90 * scale;
	editorStyle->ProjectSpaceBetweenCells = 15 * scale;
	editorStyle->ProjectCellIconPadding = 8 * scale;
	editorStyle->ProjectExpandIconSize = 16 * scale;
	editorStyle->ProjectFolderIconSize = 16 * scale;
	editorStyle->InspectorIndent = 20 * scale;
}

void ImGui::LoadDefaultEditorFonts()
{
	ImFontConfig cfg = {};
	cfg.FontBuilderFlags |= ImGuiFreeTypeBuilderFlags::ImGuiFreeTypeBuilderFlags_LightHinting;

	ImGuiIO& io = ImGui::GetIO();
	float fontSize = 16 * Blueberry::Screen::GetScale();
	io.Fonts->FontBuilderIO = ImGuiFreeType::GetBuilderForFreeType();
	io.FontDefault = io.Fonts->AddFontFromFileTTF("assets/fonts/segoeui/segoeui.ttf", fontSize, &cfg, io.Fonts->GetGlyphRangesCyrillic());
}