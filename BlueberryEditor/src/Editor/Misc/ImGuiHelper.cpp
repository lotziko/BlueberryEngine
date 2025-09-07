#include "ImGuiHelper.h"

#include "Blueberry\Core\Screen.h"
#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Core\ClassDB.h"

#include "Editor\Panels\Picking\ObjectPicker.h"
#include "Editor\Serialization\SerializedProperty.h"

#include <imgui\imgui_internal.h>
#include <imgui\misc\freetype\imgui_freetype.h>
#include <imgui\misc\cpp\imgui_stdlib.h>

ImGui::EditorContext* ImGui::GEditor = NULL;
static Blueberry::List<bool> s_ChangeStack = {};
static bool s_MixedValue = false;
static const bool* s_MixedValueMask = {};

void ImGui::CreateEditorContext()
{
	GEditor = static_cast<EditorContext*>(BB_MALLOC(sizeof(ImGui::EditorContext)));
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

bool ImGui::Property(Blueberry::SerializedProperty* property, const char* label)
{
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
		std::string value(property->GetString().data());
		if (ImGui::InputText(label, &value)) // TODO mixed
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
	default:
		ImGui::Text(label);
		break;
	}
	s_MixedValue = false;
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

#define PROPERTY_LABEL( text )\
ImGui::PushID(text);\
float availableWidth = ImGui::GetContentRegionAvail().x;\
float labelWidth = std::max(150.0f, availableWidth * 0.4f);\
float valueWidth = std::max(0.0f, availableWidth - labelWidth);\
ImGui::Text(text);\
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

bool ImGui::DragVector2(const char* label, Blueberry::Vector2* v)
{
	PROPERTY_LABEL(label)
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
	PROPERTY_LABEL(label)
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
	PROPERTY_LABEL(label)
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

	bool result = false;
	ImGui::BeginGroup();
	ImGui::PushMultiItemsWidths(components, ImGui::CalcItemWidth());
	size_t typeSize = ImGui::DataTypeGetInfo(dataType)->Size;
	for (int i = 0; i < components; i++)
	{
		if (i > 0) ImGui::SameLine();
		ImGui::PushID(i);
		ImGui::PushItemFlag(ImGuiItemFlags_MixedValue, s_MixedValueMask[i]);
		if (ImGui::DragScalar("", dataType, data, 0.1f, 0, 0, "%.3f"))
		{
			result = true;
		}
		data = (void*)((char*)data + typeSize);
		ImGui::PopItemFlag();
		ImGui::PopID();
	}
	ImGui::EndGroup();
	return result;
}

bool ImGui::EnumEdit(const char* label, int* v, const Blueberry::List<Blueberry::String>* names)
{
	if (names == nullptr || names->size() == 0)
	{
		return false;
	}

	PROPERTY_LABEL(label)
	PROPERTY_BEGIN_VALUE()

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

	PROPERTY_LABEL(label)
	PROPERTY_BEGIN_VALUE()

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

	PROPERTY_END_VALUE()
	if (result)
	{
		TriggerChange();
	}
	return result;
}

bool ImGui::BoolEdit(const char* label, bool* v)
{
	PROPERTY_LABEL(label)
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
	PROPERTY_LABEL(label)
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

bool ImGui::FloatEdit(const char* label, float* v, float min, float max)
{
	PROPERTY_LABEL(label)
	PROPERTY_BEGIN_VALUE()
	
	bool result = false;
	if (ImGui::DragFloat("##float", v, 1.0f, min, max))
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
	PROPERTY_LABEL(label)
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

bool ImGui::ObjectEdit(const char* label, Blueberry::Object** v, const size_t& type)
{
	PROPERTY_LABEL(label)
	PROPERTY_BEGIN_VALUE()

	bool result = false;
	Blueberry::Object* vObj = *v;
	if (ImGui::Button((vObj != nullptr && Blueberry::ObjectDB::IsValid(vObj)) ? vObj->GetName().c_str() : "None"))
	{
		Blueberry::ObjectPicker::Open(v, type);
	}
	if (Blueberry::ObjectPicker::GetResult(v))
	{
		vObj = *v;
		if (vObj != nullptr && vObj->GetState() == Blueberry::ObjectState::AwaitingLoading && Blueberry::ObjectDB::HasGuid(vObj))
		{
			Blueberry::AssetLoader::Load(Blueberry::ObjectDB::GetGuidFromObject(vObj));
		}
		result = true;
	}

	if (ImGui::BeginDragDropTarget())
	{
		const ImGuiPayload* payload = ImGui::GetDragDropPayload();
		if (payload != nullptr && payload->IsDataType("OBJECT_ID"))
		{
			Blueberry::ObjectId* id = static_cast<Blueberry::ObjectId*>(payload->Data);
			Blueberry::Object* object = Blueberry::ObjectDB::GetObject(*id);

			if (object != nullptr && object->IsClassType(type) && ImGui::AcceptDragDropPayload("OBJECT_ID"))
			{
				if (object->GetState() == Blueberry::ObjectState::AwaitingLoading && Blueberry::ObjectDB::HasGuid(object))
				{
					Blueberry::AssetLoader::Load(Blueberry::ObjectDB::GetGuidFromObject(object));
				}

				*v = object;
				result = true;
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

bool ImGui::ObjectEdit(const char* label, Blueberry::ObjectPtr<Blueberry::Object>* v, const size_t& type)
{
	Blueberry::Object* object = v->Get();
	if (ObjectEdit(label, &object, type))
	{
		*v = object;
		return true;
	}
	return false;
}

bool ImGui::ObjectArrayEdit(const char* label, Blueberry::List<Blueberry::ObjectPtr<Blueberry::Object>>* v, const size_t& type)
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

bool ImGui::SearchInputText(const char* hint, std::string* text)
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

void ImGui::ApplyEditorDarkTheme()
{
	ImGuiStyle* style = &ImGui::GetStyle();
	ImVec4* colors = style->Colors;

	colors[ImGuiCol_Text] = ImVec4(1.000f, 1.000f, 1.000f, 1.000f);
	colors[ImGuiCol_TextDisabled] = ImVec4(0.500f, 0.500f, 0.500f, 1.000f);
	colors[ImGuiCol_WindowBg] = ImVec4(0.180f, 0.180f, 0.180f, 1.000f);
	colors[ImGuiCol_ChildBg] = ImVec4(0.280f, 0.280f, 0.280f, 0.000f);
	colors[ImGuiCol_PopupBg] = ImVec4(0.313f, 0.313f, 0.313f, 1.000f);
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
	colors[ImGuiCol_Tab] = ImVec4(0.098f, 0.098f, 0.098f, 1.000f);
	colors[ImGuiCol_TabHovered] = ImVec4(0.352f, 0.352f, 0.352f, 1.000f);
	colors[ImGuiCol_TabActive] = ImVec4(0.195f, 0.195f, 0.195f, 1.000f);
	colors[ImGuiCol_TabUnfocused] = ImVec4(0.098f, 0.098f, 0.098f, 1.000f);
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
}

void ImGui::LoadDefaultEditorFonts()
{
	ImFontConfig cfg = {};
	cfg.FontBuilderFlags |= ImGuiFreeTypeBuilderFlags::ImGuiFreeTypeBuilderFlags_LightHinting;

	ImGuiIO& io = ImGui::GetIO(); (void)io;
	float fontSize = 16 * Blueberry::Screen::GetScale();
	io.Fonts->AddFontFromFileTTF("assets/fonts/segoeui/segoeui.ttf", fontSize, &cfg, io.Fonts->GetGlyphRangesCyrillic());
	io.FontDefault = io.Fonts->AddFontFromFileTTF("assets/fonts/segoeui/segoeui.ttf", fontSize, &cfg, io.Fonts->GetGlyphRangesCyrillic());
}
