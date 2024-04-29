#include "bbpch.h"
#include "TimeMeasurement.h"

#include <chrono>

namespace Blueberry
{
	std::stack<std::chrono::high_resolution_clock::time_point> m_Start;

	void TimeMeasurement::Start()
	{
		m_Start.push(std::chrono::high_resolution_clock::now());
	}

	void TimeMeasurement::End()
	{
		std::chrono::high_resolution_clock::time_point start = m_Start.top();
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);
		m_Start.pop();
		BB_WARNING(duration.count() << " milliseconds.");
	}
}
