#pragma once

#include "Event.h"

class WindowResizeEvent : public Event
{
public:
	EVENT_DECLARATION(WindowResize)

	WindowResizeEvent(UINT width, UINT height) : m_Width(width), m_Height(height)
	{
	}

	UINT GetWidth() const { return m_Width; }
	UINT GetHeight() const { return m_Height; }

	std::string ToString() const override
	{
		std::stringstream ss;
		ss << "WindowResizeEvent: " << m_Width << ", " << m_Height;
		return ss.str();
	}
private:
	UINT m_Width; 
	UINT m_Height;
};