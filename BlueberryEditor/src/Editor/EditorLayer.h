#pragma once

#include "Blueberry\Core\Layer.h"

namespace Blueberry
{
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
		bool m_Focused = true;

		static bool s_FrameUpdateRequested;
		static bool s_AssetsRefreshRequested;
	};
}