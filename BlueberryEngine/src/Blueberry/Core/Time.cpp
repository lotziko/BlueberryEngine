#include "Blueberry\Core\Time.h"

namespace Blueberry
{
	size_t Time::s_FrameCount = 0;
	double Time::s_Time = 0;
	float Time::s_DeltaTime = 0;
	float Time::s_FixedDeltaTime = 1.0f / 60.0f;

	size_t Time::GetFrameCount()
	{
		return s_FrameCount;
	}

	float Time::GetTime()
	{
		return static_cast<float>(s_Time);
	}

	float Time::GetDeltaTime()
	{
		return s_DeltaTime;
	}

	void Time::SetDeltaTime(float deltaTime)
	{
		s_DeltaTime = deltaTime;
	}

	float Time::GetFixedDeltaTime()
	{
		return s_FixedDeltaTime;
	}

	void Time::EndFrame()
	{
		++s_FrameCount;
		s_Time += s_DeltaTime;
	}
}
