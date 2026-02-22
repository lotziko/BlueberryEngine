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

		void AddWaitFrameCallback(const std::function<void()>& waitFrameCallback);
		void RemoveWaitFrameCallback();

		static Application* GetInstance();

		static bool IsRunning();
		static void SetRunning(const bool& isRunning);
		
	private:
		bool ProcessMessages();
		void Update();
		void Draw();

	private:
		static inline Application* s_Instance = nullptr;
		static inline bool s_IsRunning = false;
		Window* m_Window;
		LayerStack* m_LayerStack;
		std::function<void()> m_WaitFrameCallback = nullptr;
	};
}