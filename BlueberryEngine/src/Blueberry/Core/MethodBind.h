#pragma once
#include "Variant.h"

namespace Blueberry
{
	class Object;

	template <class ObjectType, class...Args>
	class MethodBindArgs;

	template <class ObjectType, class ReturnType, class...Args>
	class MethodBindReturnArgs;

	template <class ObjectType, class ReturnType>
	class MethodBindReturnNoArgs;

	class MethodBind
	{
	public:
		virtual Variant Invoke(Object* target, Variant** args) const = 0;

		template<class ObjectType, class...Args>
		static MethodBind* Create(void(ObjectType::*method)(Args...))
		{
			return new MethodBindArgs<ObjectType, Args...>(method);
		}

		template<class ObjectType, class ReturnType, class...Args>
		static MethodBind* Create(ReturnType(ObjectType::*method)(Args...))
		{
			return new MethodBindReturnArgs<ObjectType, ReturnType, Args...>(method);
		}

		template<class ObjectType, class ReturnType>
		static MethodBind* Create(ReturnType(ObjectType::*method)())
		{
			return new MethodBindReturnNoArgs<ObjectType, ReturnType>(method);
		}
	};

	template <class ObjectType, class...Args>
	class MethodBindArgs : public MethodBind
	{
	public:
		MethodBindArgs(void(ObjectType::*method)(Args...))
		{
			m_Method = method;
		}

		virtual Variant Invoke(Object* target, Variant** args) const override
		{
			(static_cast<ObjectType*>(target)->*m_Method)(*args[0]);
			return Variant();
		}
	private:
		void(ObjectType::*m_Method)(Args...);
	};

	template <class ObjectType, class ReturnType, class...Args>
	class MethodBindReturnArgs : public MethodBind
	{
	public:
		MethodBindReturnArgs(ReturnType(ObjectType::*method)(Args...))
		{
			m_Method = method;
		}

		virtual Variant Invoke(Object* target, Variant** args) const override
		{
			return (static_cast<ObjectType*>(target)->*m_Method)(*args[0]);
		}
	private:
		ReturnType(ObjectType::*m_Method)(Args...);
	};

	template <class ObjectType, class ReturnType>
	class MethodBindReturnNoArgs : public MethodBind
	{
	public:
		MethodBindReturnNoArgs(ReturnType(ObjectType::*method)())
		{
			m_Method = method;
		}

		virtual Variant Invoke(Object* target, Variant** args) const override
		{
			return (static_cast<ObjectType*>(target)->*m_Method)();
		}
	private:
		ReturnType(ObjectType::*m_Method)();
	};
}