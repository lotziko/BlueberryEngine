#include "bbpch.h"
#include "Event.h"

namespace Blueberry
{
	std::map<EventType, std::vector<EventCallback>> EventDispatcher::m_Observers = std::map<EventType, std::vector<EventCallback>>();

	void EventDispatcher::AddCallback(const EventType& type, EventCallback&& callback)
	{
		m_Observers[type].emplace_back(callback);
	}

	void EventDispatcher::Invoke(Event& event)
	{
		EventType type = event.GetEventType();

		if (m_Observers.find(type) == m_Observers.end())
			return;

		auto&& observers = m_Observers.at(type);

		for (auto&& observer : observers)
			observer(event);
	}
}
