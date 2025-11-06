#pragma once

#include "Blueberry\Core\Base.h"
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

	class MouseMoveEventArgs
	{
	public:
		MouseMoveEventArgs(float x, float y, float deltaX, float deltaY) : m_Position(x, y), m_Delta(deltaX, deltaY)
		{
		}

		const Vector2& GetPosition() const;
		const Vector2& GetDelta() const;

	private:
		Vector2 m_Position;
		Vector2 m_Delta;
	};

	using KeyDownEvent = Event<KeyEventArgs>;
	using KeyUpEvent = Event<KeyEventArgs>;
	using KeyTypeEvent = Event<KeyEventArgs>;
	using MouseMoveEvent = Event<MouseMoveEventArgs>;

	class InputEvents
	{
	public:
		static KeyDownEvent& GetKeyDown();
		static KeyUpEvent& GetKeyUp();
		static KeyTypeEvent& GetKeyTyped();
		static MouseMoveEvent& GetMouseMoved();

	private:
		static KeyDownEvent s_KeyDown;
		static KeyUpEvent s_KeyUp;
		static KeyTypeEvent s_KeyTyped;
		static MouseMoveEvent s_MouseMoved;
	};
}