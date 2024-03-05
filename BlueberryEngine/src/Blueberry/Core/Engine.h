#pragma once

#include "Blueberry\Core\LayerStack.h"
#include "Blueberry\Core\Window.h"

namespace Blueberry
{
	class Engine
	{
	public:
		bool Initialize(const WindowProperties& properties);
		void Shutdown();
		bool ProcessMessages();
		void Update();
		void Draw();

		void PushLayer(Layer* layer);

	private:
		Window* m_Window;
		LayerStack* m_LayerStack;
	};
}