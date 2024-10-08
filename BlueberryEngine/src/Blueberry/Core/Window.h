#pragma once

#include "Blueberry\Events\Event.h"

struct WindowProperties
{
	std::string Title;
	int Width;
	int Height;
	void* Data;

	WindowProperties(std::string title, int width, int height, void* data) : Title(title), Width(width), Height(height), Data(data)
	{
	}
};

class Window
{
public:
	virtual ~Window() = default;

	virtual bool ProcessMessages() = 0;

	virtual void* GetHandle() = 0;
	virtual void SetEventDispatcher(const Ref<EventDispatcher>& eventDispatcher) { m_EventDispatcher = eventDispatcher; }

	virtual int GetWidth() const = 0;
	virtual int GetHeight() const = 0;

	static Scope<Window> Create(const WindowProperties& properties);

protected:
	Ref<EventDispatcher> m_EventDispatcher;
};