#include "bbpch.h"
#include "Engine.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\RegisterSceneTypes.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\Renderer2D.h"
#include "Blueberry\Graphics\DefaultRenderer.h"
#include "Blueberry\Input\Input.h"
#include "Blueberry\Threading\JobSystem.h"
#include "Blueberry\Graphics\RegisterGraphicsTypes.h"

#include "Blueberry\Core\Layer.h"

#include <chrono>
#include <thread>

namespace Blueberry
{
	bool Engine::Initialize(const WindowProperties& properties)
	{
		m_Window = Window::Create(properties);
		m_LayerStack = new LayerStack();
		s_Instance = this;

		if (!GfxDevice::Initialize(properties.Width, properties.Height, m_Window->GetHandle()))
		{
			return false;
		}

		if (!Renderer2D::Initialize())
		{
			return false;
		}

		DefaultRenderer::Initialize();
		Input::Initialize();
		JobSystem::Initialize();

		RegisterSceneTypes();
		RegisterGraphicsTypes();

		return true;
	}

	void Engine::Shutdown()
	{
		Renderer2D::Shutdown();
		GfxDevice::Shutdown();
		DefaultRenderer::Shutdown();
		Input::Shutdown();
		delete m_LayerStack;
		delete m_Window;
	}

	void Engine::Run()
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

	void Engine::PushLayer(Layer* layer)
	{
		m_LayerStack->PushLayer(layer);
		layer->OnAttach();
	}

	void Engine::AddWaitFrameCallback(void(*waitFrameCallback)())
	{
		m_WaitFrameCallback = waitFrameCallback;
	}

	void Engine::RemoveWaitFrameCallback(void(*waitFrameCallback)())
	{
		if (m_WaitFrameCallback == waitFrameCallback)
		{
			m_WaitFrameCallback = nullptr;
		}
	}

	Engine* Engine::GetInstance()
	{
		return s_Instance;
	}

	bool Engine::ProcessMessages()
	{
		return m_Window->ProcessMessages();
	}

	void Engine::Update()
	{
		if (m_Window->IsActive())
		{
			for (Layer* layer : *m_LayerStack)
			{
				layer->OnUpdate();
			}
		}
	}

	void Engine::Draw()
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