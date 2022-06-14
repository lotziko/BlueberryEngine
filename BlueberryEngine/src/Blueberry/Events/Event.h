#pragma once

#include "Blueberry\Core\Base.h"
#include <functional>
#include <map>

namespace Blueberry
{
	enum class EventType
	{
		None = 0,
		WindowClose, WindowResize, WindowFocus, WindowLostFocus, WindowMoved,
		KeyPressed, KeyReleased, KeyTyped,
		MouseButtonPressed, MouseButtonReleased, MouseMoved, MouseScrolled
	};

#define EVENT_DECLARATION( type )															\
	static EventType GetStaticType() { return EventType::type; }							\
	virtual EventType GetEventType() const override { return GetStaticType(); }				\
	virtual const char* GetName() const override { return #type; }							\

	class Event
	{
	public:
		virtual ~Event() = default;
		virtual EventType GetEventType() const = 0;
		virtual const char* GetName() const = 0;
		virtual std::string ToString() const { return GetName(); }
	};

	using EventCallback = std::function<void(const Event&)>;

	//#define BIND_EVENT(fn) [this](auto&&... args) -> decltype(auto) { return this->fn(std::forward<decltype(args)>(args)...); }
#define BIND_EVENT(fn) std::bind(&fn, this, std::placeholders::_1)

	class EventDispatcher
	{
	public:
		void AddCallback(const EventType& type, EventCallback&& callback);
		void Invoke(Event& event) const;

	private:
		std::map<EventType, std::vector<EventCallback>> m_Observers;
	};

	inline void EventDispatcher::AddCallback(const EventType& type, EventCallback&& callback)
	{
		m_Observers[type].emplace_back(callback);
	}

	inline void EventDispatcher::Invoke(Event& event) const
	{
		EventType type = event.GetEventType();

		if (m_Observers.find(type) == m_Observers.end())
			return;

		auto&& observers = m_Observers.at(type);

		for (auto&& observer : observers)
			observer(event);
	}
}