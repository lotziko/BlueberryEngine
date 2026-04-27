#pragma once

#include "Blueberry\Core\Layer.h"

namespace Blueberry
{
	class Scene;
	class WindowResizeEventArgs;

	class RuntimeLayer : public Layer
	{
	public:
		RuntimeLayer() = default;

		virtual void OnAttach() override;
		virtual void OnDetach() override;
		virtual void OnUpdate() override;
		virtual void OnDraw() override;

		void OnWindowResize(const WindowResizeEventArgs& args);
		void OnWindowFocus();
		void OnWindowUnfocus();

	private:
		Scene* m_Scene = nullptr;
	};
}