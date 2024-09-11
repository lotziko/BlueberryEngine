#pragma once

#include "Blueberry\Events\Event.h"

namespace Blueberry
{
	using KeyCode = unsigned char;

	class KeyEventArgs
	{
	public:
		KeyEventArgs(KeyCode keyCode) : m_KeyCode(keyCode)
		{
		}

		const KeyCode& GetKeyCode() const;

	private:
		KeyCode m_KeyCode;
	};

	using KeyDownEvent = Event<KeyEventArgs>;
	using KeyUpEvent = Event<KeyEventArgs>;

	class InputEvents
	{
	public:
		static KeyDownEvent& GetKeyDown();
		static KeyUpEvent& GetKeyUp();

	private:
		static KeyDownEvent s_KeyDown;
		static KeyUpEvent s_KeyUp;
	};
}