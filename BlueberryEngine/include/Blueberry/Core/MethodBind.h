#pragma once

namespace Blueberry
{
	class MethodBind
	{
	public:
		virtual void Invoke(void* target) const = 0;

		template<class ObjectType>
		static MethodBind* Create(void(ObjectType::*method)())
		{
			return new MethodBindNoArgs<ObjectType>(method);
		}
	};

	template <class ObjectType>
	class MethodBindNoArgs : public MethodBind
	{
	public:
		MethodBindNoArgs(void(ObjectType::*method)())
		{
			m_Method = method;
		}

		virtual void Invoke(void* target) const override
		{
			(static_cast<ObjectType*>(target)->*m_Method)();
		}
	private:
		void(ObjectType::*m_Method)();
	};
}