#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Time
	{
	public:
		static const size_t GetFrameCount();
		static const float GetTime();
		static const float GetDeltaTime();
		static void IncrementFrameCount();

	private:
		static size_t m_FrameCount;
		static double m_Time;
	};
}