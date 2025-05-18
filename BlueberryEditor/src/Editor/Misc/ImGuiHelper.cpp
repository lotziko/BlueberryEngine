#include "ImGuiHelper.h"

#include "Blueberry\Core\Screen.h"
#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Assets\AssetLoader.h"

#include "Editor\Panels\Picking\ObjectPicker.h"

#include <imgui\imgui_internal.h>
#include <imgui\misc\freetype\imgui_freetype.h>
#include <imgui\misc\cpp\imgui_stdlib.h>

ImGui::EditorContext* ImGui::GEditor = NULL;

void ImGui::CreateEditorContext()
{
	GEditor = static_cast<EditorContext*>(BB_MALLOC(sizeof(ImGui::EditorContext)));
}

ImGui::EditorStyle& ImGui::GetEditorStyle()
{
	return GEditor->Style;
}

bool ImGui::DragVector2(const char* label, Blueberry::Vector2* v)
{
	ImGui::PushID(label);

	ImGui::Text(label);
	ImGui::SameLine();
	ImGui::SetCursorPosX(100);

	float vector2[2] = { v->x, v->y };
	if (ImGui::DragFloat2("##vector2", vector2, 0.1f))
	{
		v->x = vector2[0];
		v->y = vector2[1];

		ImGui::PopID();
		return true;
	}
	ImGui::PopID();
	return false;
}

bool ImGui::DragVector3(const char* label, Blueberry::Vector3* v)
{
	ImGui::PushID(label);

	ImGui::Text(label);
	ImGui::SameLine();
	ImGui::SetCursorPosX(100);

	float vector3[3] = { v->x, v->y, v->z };
	if (ImGui::DragFloat3("##vector3", vector3, 0.1f))
	{
		v->x = vector3[0];
		v->y = vector3[1];
		v->z = vector3[2];

		ImGui::PopID();
		return true;
	}
	ImGui::PopID();
	return false;
}

bool ImGui::EnumEdit(const char* label, int* v, const Blueberry::List<Blueberry::String>* names)
{
	if (names == nullptr || names->size() == 0)
	{
		return false;
	}

	ImGui::PushID(label);

	ImGui::Text(label);
	ImGui::SameLine();
	ImGui::SetCursorPosX(100);

	if (ImGui::BeginCombo("##enum", names->at(*v).c_str()))
	{
		for (int i = 0; i < names->size(); i++)
		{
			bool isSelected = *v == i;
			if (ImGui::Selectable(names->at(i).c_str(), isSelected))
			{
				*v = i;
			}

			if (isSelected)
			{
				ImGui::SetItemDefaultFocus();
			}
		}
		ImGui::EndCombo();
		ImGui::PopID();
		return true;
	}
	ImGui::PopID();
	return false;
}

bool ImGui::BoolEdit(const char* label, bool* v)
{
	ImGui::PushID(label);

	ImGui::Text(label);
	ImGui::SameLine();
	ImGui::SetCursorPosX(100);

	if (ImGui::Checkbox("##bool", v))
	{
		ImGui::PopID();
		return true;
	}
	ImGui::PopID();
	return false;
}

bool ImGui::IntEdit(const char* label, int* v)
{
	ImGui::PushID(label);

	ImGui::Text(label);
	ImGui::SameLine();
	ImGui::SetCursorPosX(100);

	if (ImGui::DragInt("##int", v))
	{
		ImGui::PopID();
		return true;
	}
	ImGui::PopID();
	return false;
}

bool ImGui::FloatEdit(const char* label, float* v, float min, float max)
{
	ImGui::PushID(label);

	ImGui::Text(label);
	ImGui::SameLine();
	ImGui::SetCursorPosX(100);

	if (ImGui::DragFloat("##float", v, 1.0f, min, max))
	{
		ImGui::PopID();
		return true;
	}
	ImGui::PopID();
	return false;
}

bool ImGui::ColorEdit(const char* label, Blueberry::Color* v)
{
	ImGui::PushID(label);

	ImGui::Text(label);
	ImGui::SameLine();
	ImGui::SetCursorPosX(100);

	float color4[4] = { v->R(), v->G(), v->B(), v->A() };
	if (ImGui::ColorEdit4("##color4", color4))
	{
		v->R(color4[0]);
		v->G(color4[1]);
		v->B(color4[2]);
		v->A(color4[3]);

		ImGui::PopID();
		return true;
	}
	ImGui::PopID();
	return false;
}

bool ImGui::ObjectEdit(const char* label, Blueberry::Object** v, const size_t& type)
{
	bool result = false;

	ImGui::PushID(label);

	ImGui::Text(label);
	ImGui::SameLine();
	ImGui::SetCursorPosX(150);

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

	ImGui::PopID();
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
