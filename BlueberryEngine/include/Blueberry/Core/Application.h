#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	struct WindowProperties;
	class Window;
	class Layer;
	class LayerStack;

	class Application
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		bool Initialize(const WindowProperties& properties);
		void Shutdown();
		void Run();

		Window* GetWindow();

		void PushLayer(Layer* layer);
		void PopLayer(Layer* layer);

		void AddWaitForFrameCallback(const std::function<void()>& waitForFrameCallback);
		void RemoveWaitForFrameCallback();

		static Application* GetInstance();

		static bool IsRunning();
		static void SetRunning(bool isRunning);
		
	private:
		bool ProcessMessages();
		void Update();
		void Draw();

	private:
		static Application* s_Instance;
		static bool s_IsRunning;
		Window* m_Window;
		LayerStack* m_LayerStack;
		std::function<void()> m_WaitForFrameCallback = nullptr;
	};
}