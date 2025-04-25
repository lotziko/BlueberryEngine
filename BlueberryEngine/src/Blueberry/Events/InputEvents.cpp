#include "bbpch.h"
#include "InputEvents.h"

namespace Blueberry
{
	KeyUpEvent InputEvents::s_KeyDown = {};
	KeyUpEvent InputEvents::s_KeyUp = {};
	KeyUpEvent InputEvents::s_KeyTyped = {};
	MouseMoveEvent InputEvents::s_MouseMoved = {};

	const KeyCode& KeyEventArgs::GetKeyCode() const
	{
		return m_KeyCode;
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