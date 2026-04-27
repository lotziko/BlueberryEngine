#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class BB_API Time
	{
	public:
		static size_t GetFrameCount();
		static float GetTime();
		static float GetDeltaTime();
		static void SetDeltaTime(float deltaTime);
		static float GetFixedDeltaTime();
		static void EndFrame();

	private:
		static size_t s_FrameCount;
		static double s_Time;
		static float s_DeltaTime;
		static float s_FixedDeltaTime;
	};
}