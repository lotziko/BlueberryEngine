#pragma once

namespace Blueberry
{
	class Variant
	{
	public:
		Variant() = default;
		
		template<class ObjectType>
		Variant(const ObjectType* data);

		template<class ObjectType>
		operator const ObjectType*();

		template<class ObjectType>
		ObjectType* Get();
		
	private:
		void* m_Data;
	};

	template<class ObjectType>
	inline Variant::Variant(const ObjectType* data)
	{
		m_Data = (void*)(data);
	}

	template<class ObjectType>
	inline Variant::operator const ObjectType*()
	{
		return static_cast<ObjectType*>(m_Data);
	}

	template<class ObjectType>
	inline ObjectType* Variant::Get()
	{
		return static_cast<ObjectType*>(m_Data);
	}
}

	