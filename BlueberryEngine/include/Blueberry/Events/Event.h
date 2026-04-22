#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Delegate.h"

namespace Blueberry
{
	template<typename... Args>
	class Event
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		Event() { m_Callbacks = std::make_shared<List<CallbackData>>(); }

		template <class OwnerObject, void(OwnerObject::*method)(Args...)>
		void AddCallback(OwnerObject* const object);
		template <void(*method)(Args...)>
		void AddCallback();
		template <class OwnerObject, void(OwnerObject::*method)(Args...)>
		void RemoveCallback(OwnerObject* const object);
		template <void(*method)(Args...)>
		void RemoveCallback();
		bool HasCallbacks();
		void Invoke(Args... args);

	private:
		struct CallbackData
		{
			uint64_t ownerPtr;
			uint64_t methodPtr;
			Delegate<Args...> delegate;
		};

		std::shared_ptr<List<CallbackData>> m_Callbacks;
		bool m_IsInvoking = false;
	};

	template <typename... Args>
	template <class OwnerObject, void(OwnerObject::*method)(Args...)>
	inline void Event<Args...>::AddCallback(OwnerObject* const object)
	{
		CallbackData data = { reinterpret_cast<uint64_t>(object), *reinterpret_cast<uint64_t*>(&method), Delegate<Args...>::Create<OwnerObject, method>(object) };
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

	template <typename... Args>
	template <void(*method)(Args...)>
	inline void Event<Args...>::AddCallback()
	{
		CallbackData data = { 0, *reinterpret_cast<uint64_t*>(&method), Delegate<Args...>::Create<method>() };
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

	template <typename... Args>
	template <class OwnerObject, void(OwnerObject::*method)(Args...)>
	inline void Event<Args...>::RemoveCallback(OwnerObject* const object)
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

	template <typename... Args>
	template <void(*method)(Args...)>
	inline void Event<Args...>::RemoveCallback()
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

	template<typename... Args>
	inline bool Event<Args...>::HasCallbacks()
	{
		return m_Callbacks->size() > 0;
	}

	template<typename... Args>
	inline void Event<Args...>::Invoke(Args... args)
	{
		m_IsInvoking = true;
		auto snapshot = m_Callbacks;
		for (auto& callback : *snapshot)
		{
			callback.delegate.Invoke(std::forward<Args>(args)...);
		}
		m_IsInvoking = false;
	}
}