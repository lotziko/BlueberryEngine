#include "EditorLayer.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Core\ServiceContainer.h"
#include "Blueberry\Content\ContentManager.h"
#include "Blueberry\Graphics\SceneRenderer.h"

#include "Editor\Misc\ImGuiHelper.h"

#include "imgui\imgui.h"
#include "imgui\imgui_internal.h"

void EditorLayer::OnAttach()
{
	m_Scene = CreateRef<Scene>();
	m_Scene->Initialize();

	ServiceContainer::ContentManager->Load<Texture>("assets/TestImage.png", m_BackgroundTexture);

	auto mainCamera = m_Scene->CreateEntity("Camera");
	mainCamera->AddComponent<Camera>();
	mainCamera->GetTransform()->SetLocalPosition(Vector3(0, 0, 0));

	auto test = m_Scene->CreateEntity("Test");
	test->AddComponent<SpriteRenderer>();
	auto sprite = test->GetComponent<SpriteRenderer>();
	sprite->SetTexture(m_BackgroundTexture);

	m_SceneHierarchy = SceneHierarchy(m_Scene);
	m_SceneInspector = SceneInspector();

	m_SceneArea = SceneArea(m_Scene);
	m_SceneArea.SetCamera(mainCamera->GetComponent<Camera>());

	if (ServiceContainer::GraphicsDevice->CreateImGuiRenderer(m_ImGuiRenderer))
	{
		ImGuiIO& io = ImGui::GetIO(); (void)io;
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

		ImGui::ApplyEditorDarkTheme();
		ImGui::LoadDefaultEditorFonts();
	}

	ServiceContainer::EventDispatcher->AddCallback(EventType::WindowResize, BIND_EVENT(EditorLayer::OnResizeEvent));
}

void EditorLayer::OnDraw()
{
	m_SceneArea.Draw();
	
	m_ImGuiRenderer->Begin();
	DrawDockSpace();
	m_ImGuiRenderer->End();

	ServiceContainer::GraphicsDevice->SwapBuffers();
}

void EditorLayer::OnResizeEvent(const Event& event)
{
	auto resizeEvent = static_cast<const WindowResizeEvent&>(event);
	ServiceContainer::GraphicsDevice->ResizeBackbuffer(resizeEvent.GetWidth(), resizeEvent.GetHeight());
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

			auto centeralNode = ImGui::DockBuilderGetCentralNode(dockspace_id);
			m_SceneArea.SetViewport(Rect(centeralNode->Pos.x, centeralNode->Pos.y, centeralNode->Size.x, centeralNode->Size.y));
		}

		DrawMenuBar();

		m_SceneHierarchy.DrawUI();
		m_SceneInspector.DrawUI();

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
