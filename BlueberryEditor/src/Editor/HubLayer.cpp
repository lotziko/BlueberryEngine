#include "HubLayer.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\ImGuiRenderer.h"
#include "Blueberry\Core\Application.h"
#include "Blueberry\Events\WindowEvents.h"

#include "Editor\Misc\ImGuiHelper.h"
#include "Editor\Misc\PlatformHelper.h"
#include "Editor\ProjectCache.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	HubLayer::HubLayer(const std::function<void(Layer*, WString)>& callback) : m_Callback(callback)
	{
	}

	void HubLayer::OnAttach()
	{
		ProjectCache::Load();
		if (ImGuiRenderer::Initialize())
		{
			ImGuiIO& io = ImGui::GetIO(); (void)io;
			io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

			ImGui::CreateEditorContext();
			ImGui::ApplyEditorDarkTheme();
			ImGui::LoadDefaultEditorFonts();
		}
		
		WindowEvents::GetWindowResized().AddCallback<HubLayer, &HubLayer::OnWindowResize>(this);
	}

	void HubLayer::OnDetach()
	{
		WindowEvents::GetWindowResized().RemoveCallback<HubLayer, &HubLayer::OnWindowResize>(this);
	}

	void HubLayer::OnDraw()
	{
		GfxDevice::ClearColor({ 0, 0, 0, 1 });
		ImGuiRenderer::Begin();
		DrawHub();
		ImGuiRenderer::End();
		GfxDevice::SwapBuffers();
	}

	void HubLayer::OnWindowResize(const WindowResizeEventArgs& args)
	{
		GfxDevice::ResizeBackbuffer(args.GetWidth(), args.GetHeight());
	}

	void HubLayer::DrawHub()
	{
		ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0.0f);
		ImGui::Begin("Hub", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
		ImGui::SetWindowPos({ 0, 0 });
		ImGui::SetWindowSize(ImGui::GetIO().DisplaySize);

		const float headerHeight = 64.0f;
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 10));
		ImGui::BeginChild("Header", ImVec2(0, headerHeight), ImGuiChildFlags_AlwaysUseWindowPadding);
		ImGui::Text("Projects");
		ImGui::SameLine(ImGui::GetWindowWidth() - 120);
		if (ImGui::Button("Open Project", ImVec2(110, 32)))
		{
			String path = PlatformHelper::OpenFileDialog();
			if (path.size() > 0)
			{
				ProjectCache::Add(path);
				ProjectCache::Save();
			}
		}
		ImGui::EndChild();
		ImGui::PopStyleVar();

		ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.117f, 0.117f, 0.117f, 1));
		ImGui::BeginChild("Projects");
		ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, ImVec2(0.025f, 0.5f));
		for (auto& info : ProjectCache::Get())
		{
			ImVec2 size = ImVec2(ImGui::GetContentRegionAvail().x, 48);
			bool clicked = ImGui::Selectable(info.path.c_str(), false, 0, size);
			if (clicked)
			{
				m_Callback(this, info.wpath);
			}
			if (ImGui::BeginPopupContextItem())
			{
				if (ImGui::MenuItem("Remove"))
				{
					ProjectCache::Remove(info.path);
					ProjectCache::Save();
				}
				ImGui::EndPopup();
			}
		}
		ImGui::PopStyleVar();
		ImGui::EndChild();
		ImGui::PopStyleColor();
		ImGui::End();
		ImGui::PopStyleVar();
		ImGui::PopStyleVar();
	}
}
