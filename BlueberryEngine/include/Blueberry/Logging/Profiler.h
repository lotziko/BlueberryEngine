#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Profiler
	{
	public:
		static void StartFrame();
		static void BeginEvent(const char* name);
		static void EndEvent();

		static const Dictionary<const char*, float>& GetData();

	private:
		static Dictionary<const char*, float> s_Data;
	};
}

#define BB_PROFILE_FRAME() Blueberry::Profiler::StartFrame()
#define BB_PROFILE_BEGIN( name ) Blueberry::Profiler::BeginEvent(name)
#define BB_PROFILE_END() Blueberry::Profiler::EndEvent()