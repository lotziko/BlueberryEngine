#include "EditorLayer.h"

#include "Blueberry\Core\Time.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\ImGuiRenderer.h"
#include "Blueberry\Graphics\GfxRenderTexturePool.h"
#include "Blueberry\Math\Math.h"
#include "Blueberry\Events\WindowEvents.h"
#include "Blueberry\Physics\Physics.h"
#include "Blueberry\Scene\Scene.h"

#include "Editor\EditorSceneManager.h"
#include "Editor\Misc\ImGuiHelper.h"
#include "Editor\Inspector\RegisterObjectEditors.h"
#include "Editor\Panels\Scene\SceneArea.h"
#include "Editor\Panels\Game\GameView.h"

#include "Editor\RegisterEditorTypes.h"
#include "Editor\Assets\RegisterAssetImporters.h"
#include "Editor\Assets\RegisterIcons.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\AssemblyManager.h"
#include "Editor\Gizmos\Gizmos.h"
#include "Editor\Gizmos\IconRenderer.h"
#include "Editor\Menu\EditorMenuManager.h"

//#include "Blueberry\Graphics\OpenXRRenderer.h"

#include <fstream>
#include <imgui\imgui.h>
#include <imgui\imguizmo.h>
#include <imgui\imgui_internal.h>

namespace Blueberry
{
	void EditorLayer::OnAttach()
	{
		RegisterEditorTypes();
		RegisterAssetImporters();
		RegisterIcons();

		RegisterObjectEditors();
		AssemblyManager::Build(false);
		AssemblyManager::Load();

		if (ImGuiRenderer::Initialize())
		{
			ImGuiIO& io = ImGui::GetIO(); (void)io;
			io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

			ImGui::CreateEditorContext();
			ImGui::ApplyEditorDarkTheme();
			ImGui::LoadDefaultEditorFonts();
		}
		Gizmos::Initialize();
		IconRenderer::Initialize();
		Physics::Initialize();
		EditorWindow::Load();

		AssetDB::Refresh();

		WindowEvents::GetWindowResized().AddCallback<EditorLayer, &EditorLayer::OnWindowResize>(this);
		WindowEvents::GetWindowFocused().AddCallback<EditorLayer, &EditorLayer::OnWindowFocus>(this);
		WindowEvents::GetWindowUnfocused().AddCallback<EditorLayer, &EditorLayer::OnWindowUnfocus>(this);
	}

	void EditorLayer::OnDetach()
	{
		EditorWindow::Save();
		for (auto& window : EditorWindow::GetWindows())
		{
			Object::Destroy(window.Get());
		}
		Gizmos::Shutdown();
		IconRenderer::Shutdown();
		Physics::Shutdown();
		GfxRenderTexturePool::Shutdown();
		ImGuiRenderer::Shutdown();
		/*if (OpenXRRenderer::IsActive())
		{
			OpenXRRenderer::Shutdown();
		}*/
		WindowEvents::GetWindowResized().RemoveCallback<EditorLayer, &EditorLayer::OnWindowResize>(this);
		WindowEvents::GetWindowFocused().RemoveCallback<EditorLayer, &EditorLayer::OnWindowFocus>(this);
		WindowEvents::GetWindowUnfocused().RemoveCallback<EditorLayer, &EditorLayer::OnWindowUnfocus>(this);
	}

	void EditorLayer::OnUpdate()
	{
		if (m_Focused)
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
	}

	void EditorLayer::OnDraw()
	{
		if (m_Focused)
		{
			if (EditorSceneManager::IsRunning())
			{
				SceneArea::RequestRedrawAll();
			}

			GfxDevice::ClearColor({ 0, 0, 0, 1 });

			//OpenXRRenderer::BeginFrame();
			ImGuiRenderer::Begin();
			DrawMenuBar();
			//DrawTopBar();
			DrawDockSpace();
			ImGuiRenderer::End();
			//OpenXRRenderer::EndFrame();

			GfxDevice::SwapBuffers();

			if (s_FrameUpdateRequested)
			{
				Time::IncrementFrameCount();
				GfxRenderTexturePool::Update();
				s_FrameUpdateRequested = false;
			}
		}
	}

	void EditorLayer::OnWindowResize(const WindowResizeEventArgs& event)
	{
		GfxDevice::ResizeBackbuffer(event.GetWidth(), event.GetHeight());
	}

	void EditorLayer::OnWindowFocus()
	{
		AssetDB::Refresh();

		if (AssemblyManager::Build())
		{
			EditorSceneManager::Save();
			EditorSceneManager::Unload();
			AssemblyManager::Unload();
			AssemblyManager::Load();
			EditorSceneManager::Reload();
		}

		m_Focused = true;
	}

	void EditorLayer::OnWindowUnfocus()
	{
		m_Focused = false;
	}

	void EditorLayer::RequestFrameUpdate()
	{
		s_FrameUpdateRequested = true;
	}

	void EditorLayer::DrawMenuBar()
	{
		if (ImGui::BeginMainMenuBar())
		{
			auto& root = EditorMenuManager::GetRoot();
			/*if (ImGui::BeginMenu("Blueberry Editor"))
			{
				ImGui::EndMenu();
			}*/
			for (auto it = root.children.begin(); it < root.children.end(); ++it)
			{
				if (ImGui::BeginMenu(it->name.c_str()))
				{
					for (auto itemIt = it->children.begin(); itemIt < it->children.end(); ++itemIt)
					{
						if (ImGui::MenuItem(itemIt->name.c_str()))
						{
							itemIt->clickCallback();
						}
					}
					ImGui::EndMenu();
				}
			}

			if (EditorSceneManager::GetScene() != nullptr)
			{
				if (EditorSceneManager::IsRunning())
				{
					if (ImGui::Button("Stop"))
					{
						Physics::Disable();
						//OpenXRRenderer::Shutdown();
						EditorSceneManager::Stop();
					}
				}
				else
				{
					if (ImGui::Button("Run"))
					{
						Physics::Enable();
						//OpenXRRenderer::Initialize();
						EditorSceneManager::Run();
					}
				}
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
							Physics::Disable();
							//OpenXRRenderer::Shutdown();
							EditorSceneManager::Stop();
						}
					}
					else
					{
						if (ImGui::Button("Run"))
						{
							Physics::Enable();
							//OpenXRRenderer::Initialize();
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

			EditorWindow::Draw();

			ImGui::End();
		}
	}
}