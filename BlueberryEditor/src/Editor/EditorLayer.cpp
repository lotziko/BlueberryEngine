#include "bbpch.h"
#include "EditorLayer.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\RenderTexture.h"
#include "Blueberry\Graphics\ImGuiRenderer.h"
#include "Blueberry\Math\Math.h"
#include "Blueberry\Events\WindowEvents.h"
#include "Blueberry\Physics\Physics.h"
#include "Blueberry\Scene\Scene.h"

#include "Editor\EditorSceneManager.h"
#include "Editor\Misc\ImGuiHelper.h"
#include "Editor\Inspector\RegisterObjectInspectors.h"
#include "Editor\Panels\Scene\SceneArea.h"
#include "Editor\Panels\Game\GameView.h"

#include "imgui\imgui.h"
#include "imgui\imguizmo.h"
#include "imgui\imgui_internal.h"

#include "Editor\RegisterEditorTypes.h"
#include "Editor\Assets\RegisterAssetImporters.h"
#include "Editor\Assets\RegisterIcons.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Gizmos\Gizmos.h"
#include "Editor\Gizmos\IconRenderer.h"

#include "Blueberry\Graphics\OpenXRRenderer.h"

#include <fstream>

namespace Blueberry
{
	void EditorLayer::OnAttach()
	{
		RegisterEditorTypes();
		RegisterAssetImporters();
		RegisterIcons();
		AssetDB::Refresh();

		RegisterObjectInspectors();

		m_SceneHierarchy = new SceneHierarchy();
		m_SceneInspector = new SceneInspector();

		m_SceneArea = new SceneArea();
		m_GameView = new GameView();

		m_ProjectBrowser = new ProjectBrowser();

		if (ImGuiRenderer::Initialize())
		{
			ImGuiIO& io = ImGui::GetIO(); (void)io;
			io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

			ImGui::ApplyEditorDarkTheme();
			ImGui::LoadDefaultEditorFonts();
		}
		Gizmos::Initialize();
		IconRenderer::Initialize();
		WindowEvents::GetWindowResized().AddCallback<EditorLayer, &EditorLayer::OnWindowResize>(this);
		WindowEvents::GetWindowFocused().AddCallback<EditorLayer, &EditorLayer::OnWindowFocus>(this);
	}

	void EditorLayer::OnDetach()
	{
		delete m_SceneHierarchy;
		delete m_SceneInspector;
		delete m_SceneArea;
		delete m_GameView;
		delete m_ProjectBrowser;
		Gizmos::Shutdown();
		IconRenderer::Shutdown();
		if (OpenXRRenderer::IsActive())
		{
			OpenXRRenderer::Shutdown();
		}
		WindowEvents::GetWindowResized().RemoveCallback<EditorLayer, &EditorLayer::OnWindowResize>(this);
		WindowEvents::GetWindowFocused().RemoveCallback<EditorLayer, &EditorLayer::OnWindowFocus>(this);
	}

	void EditorLayer::OnUpdate()
	{
		if (EditorSceneManager::IsRunning())
		{
			Scene* scene = EditorSceneManager::GetScene();
			if (scene != nullptr)
			{
				Physics::Update(1.0f / 60.0f);
				scene->Update(1.0f / 60.0f);
			}
		}
	}

	void EditorLayer::OnDraw()
	{
		if (EditorSceneManager::IsRunning())
		{
			SceneArea::RequestRedrawAll();
		}

		GfxDevice::ClearColor({ 0, 0, 0, 1 });

		OpenXRRenderer::BeginFrame();
		ImGuiRenderer::Begin();
		//DrawMenuBar();
		DrawTopBar();
		DrawDockSpace();
		ImGuiRenderer::End();
		OpenXRRenderer::EndFrame();

		GfxDevice::SwapBuffers();

		if (s_FrameUpdateRequested)
		{
			Time::IncrementFrameCount();
			RenderTexture::UpdateTemporary();
			s_FrameUpdateRequested = false;
		}
	}

	void EditorLayer::OnWindowResize(const WindowResizeEventArgs& event)
	{
		GfxDevice::ResizeBackbuffer(event.GetWidth(), event.GetHeight());
	}

	void EditorLayer::OnWindowFocus()
	{
		//AssetDB::Refresh();
	}

	void EditorLayer::RequestFrameUpdate()
	{
		s_FrameUpdateRequested = true;
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

	void EditorLayer::DrawTopBar()
	{
		ImGuiViewport* viewport = ImGui::GetMainViewport();
		if (ImGui::BeginViewportSideBar("TopBar", viewport, ImGuiDir_Up, ImGui::GetFrameHeight(), ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_MenuBar)) 
		{
			if (ImGui::BeginMenuBar())
			{
				if (EditorSceneManager::GetScene() != nullptr)
				{
					if (EditorSceneManager::IsRunning())
					{
						if (ImGui::Button("Stop"))
						{
							Physics::Shutdown();
							OpenXRRenderer::Shutdown();
							EditorSceneManager::Stop();
						}
					}
					else
					{
						if (ImGui::Button("Run"))
						{
							Physics::Initialize();
							OpenXRRenderer::Initialize();
							EditorSceneManager::Run();
						}
					}
				}
				ImGui::EndMenuBar();
			}
			ImGui::End();
		}
	}

	void EditorLayer::DrawDockSpace()
	{
		//Dockspace
		{
			static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;
			ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;

			ImGuiViewport* viewport = ImGui::GetMainViewport();
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
			ImGui::Begin("DockSpace", nullptr, window_flags);
			ImGui::PopStyleVar();

			ImGui::PopStyleVar(2);

			ImGuiIO& io = ImGui::GetIO();

			if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
			{
				ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
				ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
			}

			m_SceneHierarchy->DrawUI();
			m_SceneInspector->DrawUI();
			m_GameView->DrawUI();
			m_SceneArea->DrawUI();
			m_ProjectBrowser->DrawUI();

			ImGui::End();
		}
	}
}