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
			
		static bool IsKeyDown(const KeyCode& key);
		static Vector2 GetMousePosition();
		static Vector2 GetMouseDelta();

	private:
		static void OnKeyDown(const KeyEventArgs& args);
		static void OnKeyUp(const KeyEventArgs& args);
		static void OnMouseMove(const MouseMoveEventArgs& args);

	private:
		static bool s_State[256];
		static Vector2 s_MousePosition;
		static Vector2 s_MouseDelta;
		static size_t s_DeltaFrame;
	};
}