#pragma once

#include "Event.h"

namespace Blueberry
{
	using KeyCode = unsigned char;

	class KeyEvent : public Event
	{
	public:
		KeyCode GetKeyCode() const { return m_KeyCode; }

	protected:
		KeyEvent(const KeyCode keycode) : m_KeyCode(keycode)
		{
		}

		KeyCode m_KeyCode;
	};

	class KeyPressedEvent : public KeyEvent
	{
	public:
		EVENT_DECLARATION(KeyPressed)

		KeyPressedEvent(const KeyCode keycode) : KeyEvent(keycode)
		{
		}

		std::string ToString() const override
		{
			std::stringstream ss;
			ss << "KeyPressedEvent: " << m_KeyCode;
			return ss.str();
		}
	};

	class KeyReleasedEvent : public KeyEvent
	{
	public:
		EVENT_DECLARATION(KeyReleased)

		KeyReleasedEvent(const KeyCode keycode) : KeyEvent(keycode)
		{
		}

		std::string ToString() const override
		{
			std::stringstream ss;
			ss << "KeyReleasedEvent: " << m_KeyCode;
			return ss.str();
		}
	};

	class KeyTypedEvent : public KeyEvent
	{
	public:
		EVENT_DECLARATION(KeyTyped)

		KeyTypedEvent(const KeyCode keycode) : KeyEvent(keycode)
		{
		}

		std::string ToString() const override
		{
			std::stringstream ss;
			ss << "KeyTypedEvent: " << m_KeyCode;
			return ss.str();
		}
	};
}