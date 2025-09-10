#pragma once

#include <functional>

namespace Blueberry
{
	struct JobDispatchArgs
	{
		uint32_t jobIndex;
		uint32_t groupIndex;
	};

	class JobSystem
	{
	public:
		static void Initialize();

		static void Execute(const std::function<void()>& job);
		static void Dispatch(uint32_t jobCount, uint32_t groupSize, const std::function<void(JobDispatchArgs)>& job);

		static bool IsBusy();
		static void Wait();
	};
}