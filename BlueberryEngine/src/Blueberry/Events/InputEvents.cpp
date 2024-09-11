#include "bbpch.h"
#include "InputEvents.h"

namespace Blueberry
{
	KeyUpEvent InputEvents::s_KeyDown = {};
	KeyUpEvent InputEvents::s_KeyUp = {};

	const KeyCode& KeyEventArgs::GetKeyCode() const
	{
		return m_KeyCode;
	}

	KeyDownEvent& InputEvents::GetKeyDown()
	{
		return s_KeyDown;
	}

	KeyUpEvent& InputEvents::GetKeyUp()
	{
		return s_KeyUp;
	}
}