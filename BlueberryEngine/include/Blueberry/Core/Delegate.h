#pragma once

#include <utility>

namespace Blueberry
{
	// Based on https://medium.com/@kayraurf/delegates-in-c-polos-56211a7536ba

	template<typename... Args>
	class Delegate;

	template<>
	class Delegate<void>
	{
		template<class OwnerObject, void(OwnerObject::*methodPtr)()>
		static Delegate Create(OwnerObject* const object)
		{
			Delegate delegate;
			delegate.m_Object = object;
			delegate.m_Method = &MethodStub<OwnerObject, methodPtr>;
			return delegate;
		}

		template<void(*methodPtr)()>
		static Delegate Create()
		{
			Delegate delegate;
			delegate.m_Object = 0;
			delegate.m_Method = &FunctionStub<methodPtr>;
			return delegate;
		}

		void Invoke()
		{
			(*m_Method)(m_Object);
		}

	private:
		template <class OwnerObject, void(OwnerObject::*methodPtr)()>
		static void MethodStub(void* object)
		{
			return (static_cast<OwnerObject*>(object)->*methodPtr)();
		}

		template <void(*methodPtr)()>
		static void FunctionStub(void*)
		{
			return (methodPtr)();
		}

	private:
		using StubType = void(*)(void* m_Object);

		void* m_Object;
		StubType m_Method;
	};

	template<typename... Args>
	class Delegate
	{
	public:
		template<class OwnerObject, void(OwnerObject::*methodPtr)(Args...)>
		static Delegate Create(OwnerObject* const object)
		{
			Delegate delegate;
			delegate.m_Object = object;
			delegate.m_Method = &MethodStub<OwnerObject, methodPtr>;
			return delegate;
		}

		template<void(*methodPtr)(Args...)>
		static Delegate Create()
		{
			Delegate delegate;
			delegate.m_Object = 0;
			delegate.m_Method = &FunctionStub<methodPtr>;
			return delegate;
		}

		void Invoke(Args&&... args)
		{
			(*m_Method)(m_Object, std::forward<Args>(args)...);
		}

	private:
		template <class OwnerObject, void(OwnerObject::*methodPtr)(Args...)>
		static void MethodStub(void* object, Args&&... args)
		{
			return (static_cast<OwnerObject*>(object)->*methodPtr)(std::forward<Args>(args)...);
		}

		template <void(*methodPtr)(Args...)>
		static void FunctionStub(void*, Args&&... args)
		{
			return (*methodPtr)(std::forward<Args>(args)...);
		}

	private:
		using StubType = void(*)(void* m_Object, Args&&...);

		void* m_Object;
		StubType m_Method;
	};
}