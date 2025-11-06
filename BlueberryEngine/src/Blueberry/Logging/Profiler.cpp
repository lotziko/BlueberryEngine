#include "Blueberry\Logging\Profiler.h"

#include <chrono>

namespace Blueberry
{
	Dictionary<const char*, float> Profiler::s_Data = {};
	std::stack<std::pair<const char*, std::chrono::high_resolution_clock::time_point>> s_Start;

	void Profiler::StartFrame()
	{
	}

	void Profiler::BeginEvent(const char* name)
	{
		s_Start.push(std::make_pair(name, std::chrono::high_resolution_clock::now()));
	}

	void Profiler::EndEvent()
	{
		auto pair = s_Start.top();
		const char* name = pair.first;
		std::chrono::high_resolution_clock::time_point start = pair.second;
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		std::chrono::nanoseconds duration = std::chrono::duration_cast<std::chrono::nanoseconds> (end - start);
		s_Start.pop();
		s_Data.insert_or_assign(name, duration.count() / 1000000.0f);
	}

	const Dictionary<const char*, float>& Profiler::GetData()
	{
		return s_Data;
	}
}
