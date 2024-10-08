#include "bbpch.h"
#include "Engine.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Graphics\GraphicsDevice.h"
#include "Blueberry\Graphics\Renderer2D.h"
#include "Blueberry\Core\ServiceContainer.h"

#include "Blueberry\Core\Layer.h"

bool Engine::Initialize(const WindowProperties& properties)
{
	m_Window = Window::Create(properties);

	Ref<EventDispatcher> eventDispatcher = CreateRef<EventDispatcher>();
	m_Window->SetEventDispatcher(eventDispatcher);

	Ref<GraphicsDevice> graphicsDevice = GraphicsDevice::Create();
	if (!graphicsDevice->Initialize(properties.Width, properties.Height, m_Window->GetHandle()))
	{
		return false;
	}

	Ref<Renderer2D> renderer2D = CreateRef<Renderer2D>(graphicsDevice);
	if (!renderer2D->Initialize())
	{
		return false;
	}

	Ref<ContentManager> contentManager = CreateRef<ContentManager>(graphicsDevice);

	m_ServiceContainer = CreateRef<ServiceContainer>();
	m_ServiceContainer->EventDispatcher = eventDispatcher;
	m_ServiceContainer->ContentManager = contentManager;
	m_ServiceContainer->GraphicsDevice = graphicsDevice;
	m_ServiceContainer->Renderer2D = renderer2D;

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
	layer->SetServiceContainer(m_ServiceContainer);
	layer->OnAttach();
}
