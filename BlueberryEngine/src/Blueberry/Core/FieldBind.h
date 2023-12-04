#pragma once
#include "Variant.h"

namespace Blueberry
{
	class Object;

	template <class ObjectType, class FieldType>
	class FieldBindGeneric;

	class FieldBind
	{
	public:
		virtual void Get(Object* target, Variant& variant) const = 0;
		virtual void Set(Object* target, Variant& variant) const = 0;

		template<class ObjectType, class FieldType>
		static FieldBind* Create(FieldType ObjectType::*field)
		{
			return new FieldBindGeneric<ObjectType, FieldType>(field);
		}
	};

	template<class ObjectType, class FieldType>
	class FieldBindGeneric : public FieldBind
	{
	public:
		FieldBindGeneric(FieldType ObjectType::*field)
		{
			m_Field = field;
		}

		virtual void Get(Object* target, Variant& variant) const override;
		virtual void Set(Object* target, Variant& variant) const override;

	private:
		FieldType ObjectType::*m_Field;
	};

	template<class ObjectType, class FieldType>
	inline void FieldBindGeneric<ObjectType, FieldType>::Get(Object* target, Variant& variant) const
	{
		variant = ((static_cast<ObjectType*>(target))->*m_Field);
	}

	template<class ObjectType, class FieldType>
	inline void FieldBindGeneric<ObjectType, FieldType>::Set(Object* target, Variant& variant) const
	{
		if constexpr (std::is_base_of<Object, typename std::remove_pointer<FieldType>::type>::value)
		{
			(static_cast<ObjectType*>(target))->*m_Field = static_cast<ObjectType*>(variant.operator Object*());
		}
		else
		{
			(static_cast<ObjectType*>(target))->*m_Field = variant;
		}
	}
}