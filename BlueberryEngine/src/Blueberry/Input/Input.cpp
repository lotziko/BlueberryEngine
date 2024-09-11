#include "bbpch.h"
#include "Input.h"

namespace Blueberry
{
	void Input::Initialize()
	{
		InputEvents::GetKeyDown().AddCallback<&Input::OnKeyDown>();
		InputEvents::GetKeyUp().AddCallback<&Input::OnKeyUp>();
	}

	void Input::Shutdown()
	{
		InputEvents::GetKeyDown().RemoveCallback<&Input::OnKeyDown>();
		InputEvents::GetKeyUp().RemoveCallback<&Input::OnKeyUp>();
	}

	bool Input::IsDown(const KeyCode& key)
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
}
