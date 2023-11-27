#include "bbpch.h"
#include "EditorLayer.h"

#include "Blueberry\Core\GlobalServices.h"
#include "Blueberry\Graphics\SceneRenderer.h"
#include "Blueberry\Graphics\ImGuiRenderer.h"
#include "Blueberry\Math\Math.h"
#include "Blueberry\Scene\Scene.h"

#include "Editor\Misc\ImGuiHelper.h"
#include "Editor\Inspector\RegisterObjectInspectors.h"

#include "imgui\imgui.h"
#include "imgui\imgui_internal.h"

#include "Editor\Serialization\RegisterAssetImporters.h"
#include "Editor\Serialization\AssetDB.h"

#include <fstream>

namespace Blueberry
{
	void EditorLayer::OnAttach()
	{
		RegisterAssetImporters();
		AssetDB::ImportAll();

		RegisterObjectInspectors();

		m_Scene = CreateRef<Scene>();
		m_Scene->Initialize();

		auto test = m_Scene->CreateEntity("Test");
		test->AddComponent<SpriteRenderer>();

		ryml::Tree tree;
		SerializationContext context;
		context.tree = tree;
		ryml::NodeRef root = tree.rootref();
		root |= ryml::MAP;
		m_Scene->Serialize(context, root);
		YamlHelper::Save(tree, "Test.yaml");

		m_SceneHierarchy = SceneHierarchy(m_Scene);
		m_SceneInspector = SceneInspector();

		m_SceneArea = SceneArea(m_Scene);

		m_ProjectBrowser = ProjectBrowser();

		if (g_GraphicsDevice->CreateImGuiRenderer(m_ImGuiRenderer))
		{
			ImGuiIO& io = ImGui::GetIO(); (void)io;
			io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

			ImGui::ApplyEditorDarkTheme();
			ImGui::LoadDefaultEditorFonts();
		}

		g_EventDispatcher->AddCallback(EventType::WindowResize, BIND_EVENT(EditorLayer::OnResizeEvent));
	}

	void EditorLayer::OnDraw()
	{
		g_GraphicsDevice->ClearColor({ 0, 0, 0, 1 });

		m_ImGuiRenderer->Begin();
		DrawDockSpace();
		m_ImGuiRenderer->End();

		g_GraphicsDevice->SwapBuffers();
	}

	void EditorLayer::OnResizeEvent(const Event& event)
	{
		auto resizeEvent = static_cast<const WindowResizeEvent&>(event);
		g_GraphicsDevice->ResizeBackbuffer(resizeEvent.GetWidth(), resizeEvent.GetHeight());
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

			m_SceneHierarchy.DrawUI();
			m_SceneInspector.DrawUI();
			m_SceneArea.DrawUI();
			m_ProjectBrowser.DrawUI();

			ImGui::End();
		}
	}

	void EditorLayer::DrawMenuBar()
	{
		if (ImGui::BeginMainMenuBar())
		{
			if (ImGui::BeginMenu("Test"))
			{
				ImGui::MenuItem("Test", "Ctrl+O");
				ImGui::EndMenu();
			}
			ImGui::EndMainMenuBar();
		}
	}
}