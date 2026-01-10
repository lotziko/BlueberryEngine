#include "Blueberry\Core\Application.h"

#include "..\Core\LayerStack.h"
#include "Blueberry\Core\Window.h"
#include "Blueberry\Core\Layer.h"
#include "Blueberry\Scene\Scene.h"
#include "..\Scene\RegisterSceneTypes.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\Renderer2D.h"
#include "Blueberry\Graphics\Concrete\DefaultRenderer.h"
#include "Blueberry\Input\Input.h"
#include "Blueberry\Threading\JobSystem.h"
#include "..\Graphics\RegisterGraphicsTypes.h"
#include "Blueberry\Graphics\DefaultShaders.h"
#include "Blueberry\Graphics\Skinning.h"
#include "..\Animations\RegisterAnimationsTypes.h"

#include <chrono>
#include <thread>

namespace Blueberry
{
	bool Application::Initialize(const WindowProperties& properties)
	{
		m_Window = Window::Create(properties);
		m_LayerStack = new LayerStack();
		s_Instance = this;

		ObjectDB::Initialize();

		if (!GfxDevice::Initialize(properties.width, properties.height, m_Window->GetHandle()))
		{
			return false;
		}

		if (!Renderer2D::Initialize())
		{
			return false;
		}

		RegisterSceneTypes();
		RegisterGraphicsTypes();
		RegisterAnimationsTypes();

		DefaultRenderer::Initialize();
		Input::Initialize();
		JobSystem::Initialize();
		DefaultShaders::Initialize();
		Skinning::Initialize();

		return true;
	}

	void Application::Shutdown()
	{
		Renderer2D::Shutdown();
		DefaultRenderer::Shutdown();
		Skinning::Shutdown();
		Input::Shutdown();
		delete m_LayerStack;
		delete m_Window;

		ObjectDB::Shutdown();
		GfxDevice::Shutdown();
	}

	void Application::Run()
	{
		// Based on https://stackoverflow.com/questions/63429337/limit-fps-in-loop-c
		using framerate = std::chrono::duration<int, std::ratio<1, 60>>;
		auto prev = std::chrono::system_clock::now();
		auto next = prev + framerate{ 1 };
		int N = 0;
		std::chrono::system_clock::duration sum{ 0 };

		while (ProcessMessages())
		{
			bool hasCallbacks = m_WaitFrameCallback != nullptr;
			if (hasCallbacks)
			{
				m_WaitFrameCallback();
			}
			else
			{
				std::this_thread::sleep_until(next);
				next += framerate{ 1 };
			}

			Update();
			Draw();

			if (!hasCallbacks)
			{
				auto now = std::chrono::system_clock::now();
				sum += now - prev;
				++N;
				prev = now;
			}
		}
	}

	Window* Application::GetWindow()
	{
		return m_Window;
	}

	void Application::PushLayer(Layer* layer)
	{
		m_LayerStack->PushLayer(layer);
		layer->OnAttach();
	}

	void Application::PopLayer(Layer* layer)
	{
		m_LayerStack->PopLayer(layer);
		layer->OnDetach();
	}

	void Application::AddWaitFrameCallback(void(*waitFrameCallback)())
	{
		m_WaitFrameCallback = waitFrameCallback;
	}

	void Application::RemoveWaitFrameCallback(void(*waitFrameCallback)())
	{
		if (m_WaitFrameCallback == waitFrameCallback)
		{
			m_WaitFrameCallback = nullptr;
		}
	}

	Application* Application::GetInstance()
	{
		return s_Instance;
	}

	bool Application::ProcessMessages()
	{
		return m_Window->ProcessMessages();
	}

	void Application::Update()
	{
		if (m_Window->IsActive())
		{
			for (Layer* layer : *m_LayerStack)
			{
				layer->OnUpdate();
			}
		}
	}

	void Application::Draw()
	{
		if (m_Window->IsActive())
		{
			for (Layer* layer : *m_LayerStack)
			{
				layer->OnDraw();
			}
		}
	}
}