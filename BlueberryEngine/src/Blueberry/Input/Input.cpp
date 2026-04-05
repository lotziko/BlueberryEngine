#include "Blueberry\Input\Input.h"

#include "Blueberry\Core\Time.h"
#include "Blueberry\Events\WindowEvents.h"

namespace Blueberry
{
	std::pair<bool, size_t> Input::s_State[static_cast<uint8_t>(KeyCode::KeyCount)] = {};
	Vector2 Input::s_MousePosition = Vector2::Zero;
	Vector2 Input::s_MouseDelta = Vector2::Zero;
	size_t Input::s_DeltaFrame = 0;
	bool Input::s_IsEnabled = false;

	void Input::Initialize()
	{
		InputEvents::GetKeyDown().AddCallback<&Input::OnKeyDown>();
		InputEvents::GetKeyUp().AddCallback<&Input::OnKeyUp>();
		InputEvents::GetMouseMoved().AddCallback<&Input::OnMouseMove>();
		WindowEvents::GetWindowUnfocused().AddCallback<&Input::OnWindowUnfocus>();
	}

	void Input::Shutdown()
	{
		InputEvents::GetKeyDown().RemoveCallback<&Input::OnKeyDown>();
		InputEvents::GetKeyUp().RemoveCallback<&Input::OnKeyUp>();
		InputEvents::GetMouseMoved().RemoveCallback<&Input::OnMouseMove>();
		WindowEvents::GetWindowUnfocused().RemoveCallback<&Input::OnWindowUnfocus>();
	}

	bool Input::IsKeyDown(KeyCode key)
	{
		return s_State[static_cast<uint8_t>(key)].first;
	}

	bool Input::IsKeyPressed(KeyCode key)
	{
		auto& state = s_State[static_cast<uint8_t>(key)];
		return state.first && state.second == Time::GetFrameCount();
	}

	bool Input::IsKeyReleased(KeyCode key)
	{
		auto& state = s_State[static_cast<uint8_t>(key)];
		return !state.first && state.second == Time::GetFrameCount();
	}

	void Input::OnKeyDown(const KeyEventArgs& args)
	{
		if (s_IsEnabled)
		{
			s_State[static_cast<uint8_t>(args.GetKeyCode())] = std::make_pair(true, Time::GetFrameCount());
		}
	}

	void Input::OnKeyUp(const KeyEventArgs& args)
	{
		s_State[static_cast<uint8_t>(args.GetKeyCode())] = std::make_pair(false, Time::GetFrameCount());
	}

	void Input::OnMouseMove(const MouseMoveEventArgs& args)
	{
		s_MousePosition = args.GetPosition();
		s_MouseDelta = args.GetDelta();
		s_DeltaFrame = Time::GetFrameCount();
	}

	void Input::OnWindowUnfocus()
	{
		size_t frameCount = Time::GetFrameCount();
		for (auto& state : s_State)
		{
			if (state.first)
			{
				state.first = false;
				state.second = frameCount;
			}
		}
	}

	Vector2 Input::GetMousePosition()
	{
		return s_MousePosition;
	}

	Vector2 Input::GetMouseDelta()
	{
		if (s_DeltaFrame != Time::GetFrameCount())
		{
			return Vector2::Zero;
		}
		return s_MouseDelta;
	}

	bool Input::IsEnabled()
	{
		return s_IsEnabled;
	}

	void Input::SetEnabled(bool enabled)
	{
		s_IsEnabled = enabled;
	}
}
