#pragma once

#include "Blueberry\Core\LayerStack.h"
#include "Blueberry\Core\Window.h"
#include "Blueberry\Events\Event.h"

namespace Blueberry
{
	class Engine
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		bool Initialize(const WindowProperties& properties);
		void Shutdown();
		void Run();

		void PushLayer(Layer* layer);

		void AddWaitFrameCallback(void(*waitFrameCallback)());
		void RemoveWaitFrameCallback(void(*waitFrameCallback)());

		static Engine* GetInstance();
		
	private:
		bool ProcessMessages();
		void Update();
		void Draw();

	private:
		static inline Engine* s_Instance = nullptr;
		Window* m_Window;
		LayerStack* m_LayerStack;
		void(*m_WaitFrameCallback)() = nullptr;
	};
}