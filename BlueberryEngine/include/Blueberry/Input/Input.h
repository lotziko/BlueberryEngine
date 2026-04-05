#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Events\InputEvents.h"

namespace Blueberry
{
	class BB_API Input
	{
	public:
		static void Initialize();
		static void Shutdown();
			
		static bool IsKeyDown(KeyCode key);
		static bool IsKeyPressed(KeyCode key);
		static bool IsKeyReleased(KeyCode key);
		static Vector2 GetMousePosition();
		static Vector2 GetMouseDelta();

		static bool IsEnabled();
		static void SetEnabled(bool enabled);

	private:
		static void OnKeyDown(const KeyEventArgs& args);
		static void OnKeyUp(const KeyEventArgs& args);
		static void OnMouseMove(const MouseMoveEventArgs& args);
		static void OnWindowUnfocus();

	private:
		static std::pair<bool, size_t> s_State[static_cast<uint8_t>(KeyCode::KeyCount)];
		static Vector2 s_MousePosition;
		static Vector2 s_MouseDelta;
		static size_t s_DeltaFrame;
		static bool s_IsEnabled;
	};
}