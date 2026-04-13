#include "EditorLayer.h"

#include "Blueberry\Core\Application.h"
#include "Blueberry\Core\Time.h"
#include "Blueberry\Core\Timer.h"
#include "Blueberry\Core\EngineLayer.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\ImGuiRenderer.h"
#include "Blueberry\Graphics\GfxTexturePool.h"
#include "Blueberry\Math\Math.h"
#include "Blueberry\Events\WindowEvents.h"
#include "Blueberry\Physics\Physics.h"
#include "Blueberry\Physics\PhysicsShapeCache.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Audio\Audio.h"

#include "Editor\EditorSceneManager.h"
#include "Editor\ProjectBuilder.h"
#include "Editor\Misc\ImGuiHelper.h"
#include "Editor\Inspector\RegisterObjectEditors.h"
#include "Editor\Assets\RegisterAssetImporters.h"
#include "Editor\Panels\Scene\SceneArea.h"
#include "Editor\Panels\Game\GameView.h"
#include "Editor\Physics\EditorPhysicsShapeCache.h"

#include "Editor\RegisterEditorTypes.h"
#include "Editor\Assets\RegisterIcons.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\AssemblyManager.h"
#include "Editor\Assets\EditorAssetLoader.h"
#include "Editor\Gizmos\Gizmos.h"
#include "Editor\Gizmos\IconRenderer.h"
#include "Editor\Menu\EditorMenuManager.h"
#include "Editor\Serialization\AssemblySerializer.h"
#include "Editor\Selection.h"

//#include "Blueberry\Graphics\OpenXRRenderer.h"

#include <imgui\imgui.h>
#include <imgui\imgui_internal.h>

namespace Blueberry
{
	bool EditorLayer::s_FrameUpdateRequested = true;
	bool EditorLayer::s_AssetsRefreshRequested = false;

	void EditorLayer::OnAttach()
	{
		EngineLayer::Register();
		AssetLoader::Initialize(new EditorAssetLoader());
		EngineLayer::Initialize();
		RegisterEditorTypes();
		RegisterAssetImporters();
		RegisterObjectEditors();
		RegisterIcons();
		AssemblyManager::BuildEditor(false);
		AssemblyManager::Load();

		PhysicsShapeCache::Initialize(new EditorPhysicsShapeCache());
		AssetDB::Initialize();

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
		Audio::Initialize();
		EditorWindow::Initialize();

		WindowEvents::GetWindowResized().AddCallback<EditorLayer, &EditorLayer::OnWindowResize>(this);
		WindowEvents::GetWindowFocused().AddCallback<EditorLayer, &EditorLayer::OnWindowFocus>(this);
		WindowEvents::GetWindowUnfocused().AddCallback<EditorLayer, &EditorLayer::OnWindowUnfocus>(this);
	}

	void EditorLayer::OnDetach()
	{
		AssetDB::Shutdown();
		EditorWindow::Shutdown();
		for (auto& window : EditorWindow::GetWindows())
		{
			Object::Destroy(window.Get());
		}
		Gizmos::Shutdown();
		IconRenderer::Shutdown();
		Physics::Shutdown();
		Audio::Shutdown();
		GfxTexturePool::Shutdown();
		ImGuiRenderer::Shutdown();
		EngineLayer::Shutdown();
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
			if (Application::IsRunning())
			{
				Scene* scene = EditorSceneManager::GetScene();
				if (scene != nullptr)
				{
					scene->FixedUpdate();
					Physics::Update(Time::GetFixedDeltaTime());
				}
			}
			Audio::Update();
		}
	}

	void EditorLayer::OnDraw()
	{
		if (m_Focused)
		{
			Scene* scene = EditorSceneManager::GetScene();
			if (Application::IsRunning())
			{
				if (scene != nullptr)
				{
					scene->Update();
				}

				SceneArea::RequestRedrawAll();
			}

			GfxDevice::ClearColor({ 0, 0, 0, 1 });

			//OpenXRRenderer::BeginFrame();
			ImGuiRenderer::Begin();
			DrawMenuBar();
			DrawDockSpace();
			ImGuiRenderer::End();
			//OpenXRRenderer::EndFrame();

			GfxDevice::SwapBuffers();
			
			if (s_FrameUpdateRequested)
			{
				Time::EndFrame();
				Timer::Update();
				GfxTexturePool::Update();
				s_FrameUpdateRequested = false;
			}
			EngineLayer::Update();
		}
	}

	void EditorLayer::OnWindowResize(const WindowResizeEventArgs& args)
	{
		GfxDevice::ResizeBackbuffer(args.GetWidth(), args.GetHeight());
	}

	void EditorLayer::OnWindowFocus()
	{
		if (!Application::IsRunning())
		{
			Refresh();
		}
		else
		{
			s_AssetsRefreshRequested = true;
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
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 6));

			auto& root = EditorMenuManager::GetRoot();
			
			if (ImGui::BeginMenu("File"))
			{
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10, 6));
				ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 6));
				if (ImGui::MenuItem("Save"))
				{
					AssetDB::SaveAssets();
					AssetDB::Refresh();
				}
				ImGui::PopStyleVar(2);
				ImGui::EndMenu();
			}
			for (auto it = root.children.begin(); it < root.children.end(); ++it)
			{
				if (ImGui::BeginMenu(it->name.c_str()))
				{
					ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10, 6));
					ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 6));
					for (auto itemIt = it->children.begin(); itemIt < it->children.end(); ++itemIt)
					{
						if (ImGui::MenuItem(itemIt->name.c_str()))
						{
							itemIt->clickCallback();
						}
					}
					ImGui::PopStyleVar(2);
					ImGui::EndMenu();
				}
			}

			if (EditorSceneManager::GetScene() != nullptr)
			{
				static bool isStartingFromScene = false;
				if (Application::IsRunning())
				{
					if (ImGui::Button("Stop"))
					{
						Physics::Disable();
						//OpenXRRenderer::Shutdown();
						EditorSceneManager::Stop();
						if (isStartingFromScene)
						{
							SceneArea::Open();
						}
						if (s_AssetsRefreshRequested)
						{
							Refresh();
							s_AssetsRefreshRequested = false;
						}
					}
				}
				else
				{
					if (ImGui::Button("Run"))
					{
						if (EditorWindow::Save())
						{
							isStartingFromScene = EditorWindow::IsFocused(SceneArea::Type);
							Physics::Enable();
							//OpenXRRenderer::Initialize();
							GameView::Open();
							EditorSceneManager::Run();
						}
					}
					if (ImGui::Button("Build"))
					{
						ProjectBuilder::Build(EditorSceneManager::GetScene(), StringHelper::ToString(Path::GetBuildPath()));
					}
				}
			}
			ImGui::PopStyleVar();
			ImGui::EndMainMenuBar();
		}
	}

	void EditorLayer::DrawDockSpace()
	{
		//Dockspace
		{
			static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_NoCloseButton;
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

	void EditorLayer::Refresh()
	{
		AssetDB::Refresh();

		if (AssemblyManager::BuildEditor())
		{
			AssemblySerializer serializer = {};
			serializer.Serialize();
			AssemblyManager::Unload();
			AssemblyManager::Load();
			serializer.Deserialize();
			Selection::SetActiveObject(nullptr);
		}
	}
}