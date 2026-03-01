#include "Blueberry\Events\InputEvents.h"

namespace Blueberry
{
	KeyDownEvent InputEvents::s_KeyDown = {};
	KeyUpEvent InputEvents::s_KeyUp = {};
	KeyTypeEvent InputEvents::s_KeyTyped = {};
	MouseMoveEvent InputEvents::s_MouseMoved = {};

	const KeyCode& KeyEventArgs::GetKeyCode() const
	{
		return m_KeyCode;
	}

	const wchar_t& KeyTypeEventArgs::GetKey() const
	{
		return m_Key;
	}

	const Vector2& MouseMoveEventArgs::GetPosition() const
	{
		return m_Position;
	}

	const Vector2& MouseMoveEventArgs::GetDelta() const
	{
		return m_Delta;
	}

	KeyDownEvent& InputEvents::GetKeyDown()
	{
		return s_KeyDown;
	}

	KeyUpEvent& InputEvents::GetKeyUp()
	{
		return s_KeyUp;
	}

	KeyTypeEvent& InputEvents::GetKeyTyped()
	{
		return s_KeyTyped;
	}

	MouseMoveEvent& InputEvents::GetMouseMoved()
	{
		return s_MouseMoved;
	}
}