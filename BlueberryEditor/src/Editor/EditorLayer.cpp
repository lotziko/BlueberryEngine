#include "bbpch.h"
#include "EditorLayer.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\SceneRenderer.h"
#include "Blueberry\Graphics\ImGuiRenderer.h"
#include "Blueberry\Math\Math.h"
#include "Blueberry\Events\WindowEvents.h"

#include "Editor\Misc\ImGuiHelper.h"
#include "Editor\Inspector\RegisterObjectInspectors.h"
#include "Editor\Panels\Scene\SceneArea.h"

#include "imgui\imgui.h"
#include "imgui\imguizmo.h"
#include "imgui\imgui_internal.h"

#include "Editor\RegisterEditorTypes.h"
#include "Editor\Assets\RegisterAssetImporters.h"
#include "Editor\Assets\AssetDB.h"

#include <fstream>

namespace Blueberry
{
	void EditorLayer::OnAttach()
	{
		RegisterEditorTypes();
		RegisterAssetImporters();
		AssetDB::Refresh();

		RegisterObjectInspectors();

		m_SceneHierarchy = new SceneHierarchy();
		m_SceneInspector = new SceneInspector();

		m_SceneArea = new SceneArea();

		m_ProjectBrowser = new ProjectBrowser();

		if (GfxDevice::CreateImGuiRenderer(m_ImGuiRenderer))
		{
			ImGuiIO& io = ImGui::GetIO(); (void)io;
			io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

			ImGui::ApplyEditorDarkTheme();
			ImGui::LoadDefaultEditorFonts();
		}
		WindowEvents::GetWindowResized().AddCallback<EditorLayer, &EditorLayer::OnWindowResize>(this);
	}

	void EditorLayer::OnDetach()
	{
		delete m_SceneHierarchy;
		delete m_SceneInspector;
		delete m_SceneArea;
		delete m_ProjectBrowser;
		WindowEvents::GetWindowResized().RemoveCallback<EditorLayer, &EditorLayer::OnWindowResize>(this);
	}

	void EditorLayer::OnDraw()
	{
		GfxDevice::ClearColor({ 0, 0, 0, 1 });

		m_ImGuiRenderer->Begin();
		DrawDockSpace();
		m_ImGuiRenderer->End();

		GfxDevice::SwapBuffers();
	}

	void EditorLayer::OnWindowResize(const WindowResizeEventArgs& event)
	{
		GfxDevice::ResizeBackbuffer(event.GetWidth(), event.GetHeight());
	}

	void EditorLayer::DrawDockSpace()
	{
		//Dockspace
		{
			static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;
			ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;

			const ImGuiViewport* viewport = ImGui::GetMainViewport();
			ImGui::SetNextWindowPos(viewport->WorkPos);
			ImGui::SetNextWindowSize(viewport->WorkSize);
			ImGui::SetNextWindowViewport(viewport->ID);
			ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
			window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
			window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

			if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
				window_flags |= ImGuiWindowFlags_NoBackground;

			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
			ImGui::Begin("DockSpace Demo", nullptr, window_flags);
			ImGui::PopStyleVar();

			ImGui::PopStyleVar(2);

			ImGuiIO& io = ImGui::GetIO();

			if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
			{
				ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
				ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
			}

			DrawMenuBar();

			m_SceneHierarchy->DrawUI();
			m_SceneInspector->DrawUI();
			m_SceneArea->DrawUI();
			m_ProjectBrowser->DrawUI();

			ImGui::End();
		}
	}

	void EditorLayer::DrawMenuBar()
	{
		if (ImGui::BeginMainMenuBar())
		{
			if (ImGui::BeginMenu("Test"))
			{
				if (ImGui::MenuItem("Save"))
				{
					AssetDB::SaveAssets();
				}
				ImGui::EndMenu();
			}
			ImGui::EndMainMenuBar();
		}
	}
}