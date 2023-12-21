#pragma once

#include "Blueberry\Core\Layer.h"
#include "Blueberry\Events\WindowEvent.h"

#include "Editor\Panels\Hierarchy\SceneHierarchy.h"
#include "Editor\Panels\Inspector\SceneInspector.h"
#include "Editor\Panels\Scene\SceneArea.h"
#include "Editor\Panels\Project\ProjectBrowser.h"

namespace Blueberry
{
	class Scene;
	class Camera;
	class Texture;
	class ImGuiRenderer;

	class EditorLayer : public Layer
	{
	public:
		EditorLayer() = default;

		virtual void OnAttach() override;
		virtual void OnDraw() override;
		virtual void OnResizeEvent(const Event& event);

	private:
		void DrawDockSpace();
		void DrawMenuBar();

	private:
		Scene* m_Scene;
		ImGuiRenderer* m_ImGuiRenderer;

		SceneHierarchy m_SceneHierarchy;
		SceneInspector m_SceneInspector;
		SceneArea m_SceneArea;
		ProjectBrowser m_ProjectBrowser;
	};
}