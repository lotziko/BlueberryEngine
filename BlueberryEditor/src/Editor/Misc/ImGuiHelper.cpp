#include "bbpch.h"
#include "ImGuiHelper.h"

#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "imgui\imgui_internal.h"

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

bool ImGui::EnumEdit(const char* label, int* v, const std::vector<std::string>* names)
{
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

bool ImGui::FloatEdit(const char* label, float* v)
{
	ImGui::PushID(label);

	ImGui::Text(label);
	ImGui::SameLine();
	ImGui::SetCursorPosX(100);

	if (ImGui::DragFloat("##float", v))
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

bool ImGui::ObjectEdit(const char* label, Blueberry::Object** v, const std::size_t& type)
{
	bool result = false;

	ImGui::PushID(label);

	ImGui::Text(label);
	ImGui::SameLine();
	ImGui::SetCursorPosX(150);

	Blueberry::Object* vObj = *v;
	ImGui::Text((vObj != nullptr && Blueberry::ObjectDB::IsValid(vObj)) ? vObj->GetName().c_str() : "None");

	if (ImGui::BeginDragDropTarget())
	{
		const ImGuiPayload* payload = ImGui::GetDragDropPayload();
		if (payload != nullptr && payload->IsDataType("OBJECT_ID"))
		{
			Blueberry::ObjectId* id = (Blueberry::ObjectId*)payload->Data;
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

bool ImGui::ObjectEdit(const char* label, Blueberry::ObjectPtr<Blueberry::Object>* v, const std::size_t& type)
{
	Blueberry::Object* object = v->Get();
	if (ObjectEdit(label, &object, type))
	{
		*v = object;
		return true;
	}
	return false;
}

bool ImGui::ObjectArrayEdit(const char* label, std::vector<Blueberry::ObjectPtr<Blueberry::Object>>* v, const std::size_t& type)
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

	style->ChildRounding = 4.0f;
	style->FrameBorderSize = 1.0f;
	style->FrameRounding = 2.0f;
	style->GrabMinSize = 7.0f;
	style->PopupRounding = 2.0f;
	style->ScrollbarRounding = 12.0f;
	style->ScrollbarSize = 13.0f;
	style->TabBorderSize = 1.0f;
	style->TabRounding = 0.0f;
	style->WindowRounding = 4.0f;
	style->WindowPadding = ImVec2(0.0f, 0.0f);
}

void ImGui::LoadDefaultEditorFonts()
{
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	float fontSize = 17.0f;
	io.Fonts->AddFontFromFileTTF("assets/fonts/sourcesans/SourceSansPro-Semibold.ttf", fontSize);
	io.FontDefault = io.Fonts->AddFontFromFileTTF("assets/fonts/sourcesans/SourceSansPro-Regular.ttf", fontSize);
}
