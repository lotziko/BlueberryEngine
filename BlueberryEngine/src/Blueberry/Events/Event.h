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

#define BIND_EVENT(fn) std::bind(&fn, this, std::placeholders::_1)

	class EventDispatcher
	{
	public:
		EventDispatcher() = default;
		~EventDispatcher() = default;

	public:
		static void AddCallback(const EventType& type, EventCallback&& callback);
		static void Invoke(Event& event);

	private:
		static std::map<EventType, std::vector<EventCallback>> m_Observers;
	};
}