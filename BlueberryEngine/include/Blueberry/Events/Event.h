#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Delegate.h"

namespace Blueberry
{
	template<typename EventType = void>
	class Event;

	template<>
	class Event<void>
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		Event() { m_Callbacks = std::make_shared<List<CallbackData>>(); }

		template <class OwnerObject, void(OwnerObject::*method)()>
		void AddCallback(OwnerObject* const object);
		template <void(*method)()>
		void AddCallback();
		template <class OwnerObject, void(OwnerObject::*method)()>
		void RemoveCallback(OwnerObject* const object);
		template <void(*method)()>
		void RemoveCallback();
		bool HasCallbacks();
		void Invoke();

	private:
		struct CallbackData
		{
			uint64_t ownerPtr;
			uint64_t methodPtr;
			Delegate<> delegate;
		};

		std::shared_ptr<List<CallbackData>> m_Callbacks;
		bool m_IsInvoking = false;
	};

	template<class EventType>
	class Event
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		Event() { m_Callbacks = std::make_shared<List<CallbackData>>(); }

		template <class OwnerObject, void(OwnerObject::*method)(const EventType&)>
		void AddCallback(OwnerObject* const object);
		template <void(*method)(const EventType&)>
		void AddCallback();
		template <class OwnerObject, void(OwnerObject::*method)(const EventType&)>
		void RemoveCallback(OwnerObject* const object);
		template <void(*method)(const EventType&)>
		void RemoveCallback();
		bool HasCallbacks();
		void Invoke(EventType& event);

	private:
		struct CallbackData
		{
			uint64_t ownerPtr;
			uint64_t methodPtr;
			Delegate<const EventType&> delegate;
		};

		std::shared_ptr<List<CallbackData>> m_Callbacks;
		bool m_IsInvoking = false;
	};

	template <class OwnerObject, void(OwnerObject::*method)()>
	inline void Event<void>::AddCallback(OwnerObject* const object)
	{
		CallbackData data = { reinterpret_cast<uint64_t>(object), *reinterpret_cast<uint64_t*>(&method), Delegate<>::Create<OwnerObject, method>(object) };
		if (m_IsInvoking)
		{
			std::shared_ptr<List<CallbackData>> newCallbacks = std::make_shared<List<CallbackData>>(*m_Callbacks);
			newCallbacks->push_back(std::move(data));
			m_Callbacks = newCallbacks;
		}
		else
		{
			m_Callbacks->push_back(std::move(data));
		}
	}

	template<void(*method)()>
	inline void Event<void>::AddCallback()
	{
		CallbackData data = { 0, *reinterpret_cast<uint64_t*>(&method), Delegate<>::Create<method>() };
		if (m_IsInvoking)
		{
			std::shared_ptr<List<CallbackData>> newCallbacks = std::make_shared<List<CallbackData>>(*m_Callbacks);
			newCallbacks->push_back(std::move(data));
			m_Callbacks = newCallbacks;
		}
		else
		{
			m_Callbacks->push_back(std::move(data));
		}
	}

	template <class OwnerObject, void(OwnerObject::*method)()>
	inline void Event<void>::RemoveCallback(OwnerObject* const object)
	{
		uint64_t ownerPtr = reinterpret_cast<uint64_t>(object);
		uint64_t methodPtr = *reinterpret_cast<uint64_t*>(&method);
		if (m_IsInvoking)
		{
			std::shared_ptr<List<CallbackData>> newCallbacks = std::make_shared<List<CallbackData>>(*m_Callbacks);
			for (auto it = newCallbacks->begin(); it != newCallbacks->end(); ++it)
			{
				if (it->ownerPtr == ownerPtr && it->methodPtr == methodPtr)
				{
					newCallbacks->erase(it);
					break;
				}
			}
			m_Callbacks = newCallbacks;
		}
		else
		{
			for (auto it = m_Callbacks->begin(); it != m_Callbacks->end(); ++it)
			{
				if (it->ownerPtr == ownerPtr && it->methodPtr == methodPtr)
				{
					m_Callbacks->erase(it);
					break;
				}
			}
		}
	}

	template <void(*method)()>
	inline void Event<void>::RemoveCallback()
	{
		uint64_t methodPtr = *reinterpret_cast<uint64_t*>(&method);
		if (m_IsInvoking)
		{
			std::shared_ptr<List<CallbackData>> newCallbacks = std::make_shared<List<CallbackData>>(*m_Callbacks);
			for (auto it = newCallbacks->begin(); it != newCallbacks->end(); ++it)
			{
				if (it->methodPtr == methodPtr)
				{
					newCallbacks->erase(it);
					break;
				}
			}
			m_Callbacks = newCallbacks;
		}
		else
		{
			for (auto it = m_Callbacks->begin(); it != m_Callbacks->end(); ++it)
			{
				if (it->methodPtr == methodPtr)
				{
					m_Callbacks->erase(it);
					break;
				}
			}
		}
	}

	inline bool Event<void>::HasCallbacks()
	{
		return m_Callbacks->size() > 0;
	}

	inline void Event<void>::Invoke()
	{
		m_IsInvoking = true;
		auto snapshot = m_Callbacks;
		for (auto& callback : *snapshot)
		{
			callback.delegate.Invoke();
		}
		m_IsInvoking = false;
	}

	template <class EventType>
	template <class OwnerObject, void(OwnerObject::*method)(const EventType&)>
	inline void Event<EventType>::AddCallback(OwnerObject* const object)
	{
		CallbackData data = { reinterpret_cast<uint64_t>(object), *reinterpret_cast<uint64_t*>(&method), Delegate<const EventType&>::Create<OwnerObject, method>(object) };
		if (m_IsInvoking)
		{
			std::shared_ptr<List<CallbackData>> newCallbacks = std::make_shared<List<CallbackData>>(*m_Callbacks);
			newCallbacks->push_back(std::move(data));
			m_Callbacks = newCallbacks;
		}
		else
		{
			m_Callbacks->push_back(std::move(data));
		}
	}

	template <class EventType>
	template <void(*method)(const EventType&)>
	inline void Event<EventType>::AddCallback()
	{
		CallbackData data = { 0, *reinterpret_cast<uint64_t*>(&method), Delegate<const EventType&>::Create<method>() };
		if (m_IsInvoking)
		{
			std::shared_ptr<List<CallbackData>> newCallbacks = std::make_shared<List<CallbackData>>(*m_Callbacks);
			newCallbacks->push_back(std::move(data));
			m_Callbacks = newCallbacks;
		}
		else
		{
			m_Callbacks->push_back(std::move(data));
		}
	}

	template <class EventType>
	template <class OwnerObject, void(OwnerObject::*method)(const EventType&)>
	inline void Event<EventType>::RemoveCallback(OwnerObject* const object)
	{
		uint64_t ownerPtr = reinterpret_cast<uint64_t>(object);
		uint64_t methodPtr = *reinterpret_cast<uint64_t*>(&method);
		if (m_IsInvoking)
		{
			std::shared_ptr<List<CallbackData>> newCallbacks = std::make_shared<List<CallbackData>>(*m_Callbacks);
			for (auto it = newCallbacks->begin(); it != newCallbacks->end(); ++it)
			{
				if (it->ownerPtr == ownerPtr && it->methodPtr == methodPtr)
				{
					newCallbacks->erase(it);
					break;
				}
			}
			m_Callbacks = newCallbacks;
		}
		else
		{
			for (auto it = m_Callbacks->begin(); it != m_Callbacks->end(); ++it)
			{
				if (it->ownerPtr == ownerPtr && it->methodPtr == methodPtr)
				{
					m_Callbacks->erase(it);
					break;
				}
			}
		}
	}

	template <class EventType>
	template <void(*method)(const EventType&)>
	inline void Event<EventType>::RemoveCallback()
	{
		uint64_t methodPtr = *reinterpret_cast<uint64_t*>(&method);
		if (m_IsInvoking)
		{
			std::shared_ptr<List<CallbackData>> newCallbacks = std::make_shared<List<CallbackData>>(*m_Callbacks);
			for (auto it = newCallbacks->begin(); it != newCallbacks->end(); ++it)
			{
				if (it->methodPtr == methodPtr)
				{
					newCallbacks->erase(it);
					break;
				}
			}
			m_Callbacks = newCallbacks;
		}
		else
		{
			for (auto it = m_Callbacks->begin(); it != m_Callbacks->end(); ++it)
			{
				if (it->methodPtr == methodPtr)
				{
					m_Callbacks->erase(it);
					break;
				}
			}
		}
	}

	template<class EventType>
	inline bool Event<EventType>::HasCallbacks()
	{
		return m_Callbacks->size() > 0;
	}

	template<class EventType>
	inline void Event<EventType>::Invoke(EventType& event)
	{
		m_IsInvoking = true;
		auto snapshot = m_Callbacks;
		for (auto& callback : *snapshot)
		{
			callback.delegate.Invoke(event);
		}
		m_IsInvoking = false;
	}
}