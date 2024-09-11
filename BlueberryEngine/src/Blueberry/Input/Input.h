#pragma once

#include "Blueberry\Events\InputEvents.h"

namespace Blueberry
{
	class Input
	{
	public:
		static void Initialize();
		static void Shutdown();
			
		static bool IsKeyDown(const KeyCode& key);
		static Vector2 GetMousePosition();

	private:
		static void OnKeyDown(const KeyEventArgs& args);
		static void OnKeyUp(const KeyEventArgs& args);
		static void OnMouseMove(const MouseMoveEventArgs& args);

	private:
		static inline bool s_State[256] = {};
		static inline Vector2 s_MousePosition = Vector2::Zero;
	};
}