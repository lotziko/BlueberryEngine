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
		static Vector2 GetMouseDelta();

	private:
		static void OnKeyDown(const KeyEventArgs& args);
		static void OnKeyUp(const KeyEventArgs& args);
		static void OnMouseMove(const MouseMoveEventArgs& args);

	private:
		static inline bool s_State[256] = {};
		static inline Vector2 s_MousePosition = Vector2::Zero;
		static inline Vector2 s_MouseDelta = Vector2::Zero;
		static inline size_t s_DeltaFrame;
	};
}