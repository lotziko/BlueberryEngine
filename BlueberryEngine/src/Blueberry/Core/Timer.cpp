#include "Blueberry\Core\Timer.h"

#include "Blueberry\Core\Time.h"
#include "Blueberry\Core\ObjectDB.h"

namespace Blueberry
{
	struct TimerData
	{
		float time;
		ObjectId objectId;
		std::function<void()> callback;
	};

	static Dictionary<size_t, TimerData> s_Timers;
	static size_t s_MaxHandleId = 0;

	size_t Timer::Start(float time, Object* object, const std::function<void()>& callback)
	{
		size_t handle = ++s_MaxHandleId;
		TimerData data;
		data.time = time;
		data.objectId = object == nullptr ? 0 : object->GetObjectId();
		data.callback = callback;
		s_Timers.insert_or_assign(handle, data);
		return handle;
	}

	void Timer::Stop(size_t handle)
	{
		auto it = s_Timers.find(handle);
		if (it != s_Timers.end())
		{
			s_Timers.erase(it);
		}
	}

	void Timer::Update()
	{
		float deltaTime = Time::GetDeltaTime();
		for (auto it = s_Timers.begin(); it != s_Timers.end();)
		{
			TimerData& data = it->second;
			data.time -= deltaTime;
			if (data.time <= 0.0f)
			{
				if (data.objectId == 0 || ObjectDB::IsValid(data.objectId))
				{
					data.callback();
				}
				it = s_Timers.erase(it);
			}
			else
			{
				++it;
			}
		}
	}
}