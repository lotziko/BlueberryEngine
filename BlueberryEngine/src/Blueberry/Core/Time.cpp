#include "Blueberry\Core\Time.h"

namespace Blueberry
{
	size_t Time::m_FrameCount = 0;
	double Time::m_Time = 0;

	const size_t Time::GetFrameCount()
	{
		return m_FrameCount;
	}

	const float Time::GetTime()
	{
		return static_cast<float>(m_Time);
	}

	const float Time::GetDeltaTime()
	{
		return 1.0f / 60.0f; // TODO
	}

	void Time::IncrementFrameCount()
	{
		++m_FrameCount;
		m_Time += 1.0 / 60.0; // TODO delta time instead
	}
}
