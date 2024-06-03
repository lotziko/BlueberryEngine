#include "bbpch.h"
#include "Engine.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\RegisterSceneTypes.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\Renderer2D.h"
#include "Blueberry\Graphics\Gizmos.h"
#include "Blueberry\Graphics\RegisterGraphicsTypes.h"

#include "Blueberry\Core\Layer.h"
#include "Blueberry\Core\Time.h"

namespace Blueberry
{
	bool Engine::Initialize(const WindowProperties& properties)
	{
		m_Window = Window::Create(properties);
		m_LayerStack = new LayerStack();

		if (!GfxDevice::Initialize(properties.Width, properties.Height, m_Window->GetHandle()))
		{
			return false;
		}

		if (!Renderer2D::Initialize())
		{
			return false;
		}

		if (!Gizmos::Initialize())
		{
			return false;
		}

		RegisterSceneTypes();
		RegisterGraphicsTypes();

		return true;
	}

	void Engine::Shutdown()
	{
		Gizmos::Shutdown();
		Renderer2D::Shutdown();
		GfxDevice::Shutdown();
		delete m_LayerStack;
		delete m_Window;
	}

	bool Engine::ProcessMessages()
	{
		return m_Window->ProcessMessages();
	}

	void Engine::Update()
	{
	}

	void Engine::Draw()
	{
		Time::IncrementFrameCount();
		for (Layer* layer : *m_LayerStack)
			layer->OnDraw();
	}

	void Engine::PushLayer(Layer* layer)
	{
		m_LayerStack->PushLayer(layer);
		layer->OnAttach();
	}
}