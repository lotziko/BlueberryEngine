#include "Time.h"

namespace Blueberry
{
	size_t Time::m_FrameCount = 0;

	const size_t Time::GetFrameCount()
	{
		return m_FrameCount;
	}

	void Time::IncrementFrameCount()
	{
		++m_FrameCount;
	}
}
