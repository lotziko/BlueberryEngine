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
	class GameView;
	class WindowResizeEventArgs;

	class EditorLayer : public Layer
	{
	public:
		EditorLayer() = default;

		virtual void OnAttach() override;
		virtual void OnDetach() override;
		virtual void OnUpdate() override;
		virtual void OnDraw() override;

		void OnWindowResize(const WindowResizeEventArgs& args);
		void OnWindowFocus();
		void OnWindowUnfocus();

		static void RequestFrameUpdate();

	private:
		void DrawMenuBar();
		void DrawDockSpace();
		void Refresh();

	private:
		Scene* m_Scene = nullptr;
		bool m_Focused = true;

		static bool s_FrameUpdateRequested;
		static bool s_AssetsRefreshRequested;
	};
}