#pragma once

#include "Blueberry\Core\LayerStack.h"
#include "Blueberry\Core\Window.h"

namespace Blueberry
{
	struct ServiceContainer;

	class Engine
	{
	public:
		bool Initialize(const WindowProperties& properties);
		bool ProcessMessages();
		void Update();
		void Draw();

		void PushLayer(Layer* layer);

	private:
		Scope<Window> m_Window;
		LayerStack m_LayerStack;
	};
}