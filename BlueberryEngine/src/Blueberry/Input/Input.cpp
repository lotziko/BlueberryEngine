#include "bbpch.h"
#include "Input.h"

namespace Blueberry
{
	void Input::Initialize()
	{
		InputEvents::GetKeyDown().AddCallback<&Input::OnKeyDown>();
		InputEvents::GetKeyUp().AddCallback<&Input::OnKeyUp>();
		InputEvents::GetMouseMoved().AddCallback<&Input::OnMouseMove>();
	}

	void Input::Shutdown()
	{
		InputEvents::GetKeyDown().RemoveCallback<&Input::OnKeyDown>();
		InputEvents::GetKeyUp().RemoveCallback<&Input::OnKeyUp>();
		InputEvents::GetMouseMoved().RemoveCallback<&Input::OnMouseMove>();
	}

	bool Input::IsKeyDown(const KeyCode& key)
	{
		return s_State[key];
	}

	void Input::OnKeyDown(const KeyEventArgs& args)
	{
		s_State[args.GetKeyCode()] = true;
	}

	void Input::OnKeyUp(const KeyEventArgs& args)
	{
		s_State[args.GetKeyCode()] = false;
	}

	void Input::OnMouseMove(const MouseMoveEventArgs& args)
	{
		s_MousePosition = args.GetPosition();
	}

	Vector2 Input::GetMousePosition()
	{
		return s_MousePosition;
	}
}
