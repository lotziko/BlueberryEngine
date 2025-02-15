#pragma once

#include "Blueberry\Core\Delegate.h"
#include <map>

namespace Blueberry
{
	struct PairHash
	{
		template <class T1, class T2>
		size_t operator() (const std::pair<T1, T2> &v) const
		{
			return std::hash<T1>()(v.first) ^ std::hash<T2>()(v.second) << 1;
		}
	};

	template<typename EventType = void>
	class Event;

	template<>
	class Event<void>
	{
	public:
		template <class OwnerObject, void(OwnerObject::*methodPtr)()>
		void AddCallback(OwnerObject* const object);
		template <void(*methodPtr)()>
		void AddCallback();
		template <class OwnerObject, void(OwnerObject::*methodPtr)()>
		void RemoveCallback(OwnerObject* const object);
		template <void(*methodPtr)()>
		void RemoveCallback();
		bool HasCallbacks();
		void Invoke();

	private:
		std::unordered_map<std::pair<uint64_t, uint64_t>, Delegate<>, PairHash> m_Callbacks;
	};

	template<class EventType>
	class Event
	{
	public:
		template <class OwnerObject, void(OwnerObject::*methodPtr)(const EventType&)>
		void AddCallback(OwnerObject* const object);
		template <void(*methodPtr)(const EventType&)>
		void AddCallback();
		template <class OwnerObject, void(OwnerObject::*methodPtr)(const EventType&)>
		void RemoveCallback(OwnerObject* const object);
		template <void(*methodPtr)(const EventType&)>
		void RemoveCallback();
		bool HasCallbacks();
		void Invoke(EventType& event);

	private:
		std::unordered_map<std::pair<uint64_t, uint64_t>, Delegate<const EventType&>, PairHash> m_Callbacks;
	};

	template <class OwnerObject, void(OwnerObject::*methodPtr)()>
	inline void Event<void>::AddCallback(OwnerObject* const object)
	{
		m_Callbacks.insert_or_assign(std::make_pair((uint64_t)&methodPtr, (uint64_t)object), Delegate<>::Create<OwnerObject, methodPtr>(object));
	}

	template<void(*methodPtr)()>
	inline void Event<void>::AddCallback()
	{
		m_Callbacks.insert_or_assign(std::make_pair((uint64_t)&methodPtr, 0), Delegate<>::Create<methodPtr>());
	}

	template <class OwnerObject, void(OwnerObject::*methodPtr)()>
	inline void Event<void>::RemoveCallback(OwnerObject* const object)
	{
		m_Callbacks.erase(std::make_pair((uint64_t)&methodPtr, (uint64_t)object));
	}

	template <void(*methodPtr)()>
	inline void Event<void>::RemoveCallback()
	{
		m_Callbacks.erase(std::make_pair((uint64_t)&methodPtr, 0));
	}

	inline bool Event<void>::HasCallbacks()
	{
		return m_Callbacks.size() > 0;
	}

	inline void Event<void>::Invoke()
	{
		for (auto& callback : m_Callbacks)
			callback.second.Invoke();
	}

	template <class EventType>
	template <class OwnerObject, void(OwnerObject::*methodPtr)(const EventType&)>
	inline void Event<EventType>::AddCallback(OwnerObject* const object)
	{
		m_Callbacks.insert_or_assign(std::make_pair((uint64_t)&methodPtr, (uint64_t)object), Delegate<const EventType&>::Create<OwnerObject, methodPtr>(object));
	}

	template <class EventType>
	template <void(*methodPtr)(const EventType&)>
	inline void Event<EventType>::AddCallback()
	{
		m_Callbacks.insert_or_assign(std::make_pair((uint64_t)&methodPtr, 0), Delegate<const EventType&>::Create<methodPtr>());
	}

	template <class EventType>
	template <class OwnerObject, void(OwnerObject::*methodPtr)(const EventType&)>
	inline void Event<EventType>::RemoveCallback(OwnerObject* const object)
	{
		m_Callbacks.erase(std::make_pair((uint64_t)&methodPtr, (uint64_t)object));
	}

	template <class EventType>
	template <void(*methodPtr)(const EventType&)>
	inline void Event<EventType>::RemoveCallback()
	{
		m_Callbacks.erase(std::make_pair((uint64_t)&methodPtr, 0));
	}

	template<class EventType>
	inline bool Event<EventType>::HasCallbacks()
	{
		return m_Callbacks.size() > 0;
	}

	template<class EventType>
	inline void Event<EventType>::Invoke(EventType& event)
	{
		for (auto& callback : m_Callbacks)
			callback.second.Invoke(event);
	}
}