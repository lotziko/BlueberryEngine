#include "bbpch.h"
#include "Engine.h"

#include "Blueberry\Core\GlobalServices.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\RegisterSceneTypes.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\Renderer2D.h"
#include "Blueberry\Graphics\RegisterGraphicsTypes.h"

#include "Blueberry\Core\Layer.h"

namespace Blueberry
{
	bool Engine::Initialize(const WindowProperties& properties)
	{
		m_Window = Window::Create(properties);

		g_EventDispatcher = new EventDispatcher();

		g_GraphicsDevice = GfxDevice::Create();
		if (!g_GraphicsDevice->Initialize(properties.Width, properties.Height, m_Window->GetHandle()))
		{
			return false;
		}

		g_Renderer2D = new Renderer2D();
		if (!g_Renderer2D->Initialize())
		{
			return false;
		}

		RegisterSceneTypes();
		RegisterGraphicsTypes();

		return true;
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
		for (Layer* layer : m_LayerStack)
			layer->OnDraw();
	}

	void Engine::PushLayer(Layer* layer)
	{
		m_LayerStack.PushLayer(layer);
		layer->OnAttach();
	}
}