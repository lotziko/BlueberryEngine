#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Time
	{
	public:
		static const size_t GetFrameCount();
		static void IncrementFrameCount();

	private:
		static size_t m_FrameCount;
	};
}