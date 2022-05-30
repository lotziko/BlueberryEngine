#pragma once

#include "Blueberry\Core\LayerStack.h"
#include "Blueberry\Core\Window.h"

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
	Ref<ServiceContainer> m_ServiceContainer;
	LayerStack m_LayerStack;
};
