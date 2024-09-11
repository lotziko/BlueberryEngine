#pragma once

#include "Blueberry\Events\InputEvents.h"

namespace Blueberry
{
	class Input
	{
	public:
		static void Initialize();
		static void Shutdown();
			
		static bool IsDown(const KeyCode& key);

	private:
		static void OnKeyDown(const KeyEventArgs& args);
		static void OnKeyUp(const KeyEventArgs& args);

	private:
		static inline bool s_State[256] = {};
	};
}