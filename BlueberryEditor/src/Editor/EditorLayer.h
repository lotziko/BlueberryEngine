#pragma once

#include "Blueberry\Core\Layer.h"

#include "Editor\Panels\Hierarchy\SceneHierarchy.h"
#include "Editor\Panels\Inspector\SceneInspector.h"
#include "Editor\Panels\Project\ProjectBrowser.h"

namespace Blueberry
{
	class Scene;
	class Camera;
	class Texture;
	class SceneArea;
	class ImGuiRenderer;
	class WindowResizeEventArgs;

	class EditorLayer : public Layer
	{
	public:
		EditorLayer() = default;

		virtual void OnAttach() override;
		virtual void OnDetach() override;
		virtual void OnDraw() override;
		virtual void OnWindowResize(const WindowResizeEventArgs& event);

	private:
		void DrawDockSpace();
		void DrawMenuBar();

	private:
		Scene* m_Scene = nullptr;
		ImGuiRenderer* m_ImGuiRenderer = nullptr;

		SceneHierarchy* m_SceneHierarchy = nullptr;
		SceneInspector* m_SceneInspector = nullptr;
		SceneArea* m_SceneArea = nullptr;
		ProjectBrowser* m_ProjectBrowser = nullptr;
	};
}