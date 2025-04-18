#pragma once

namespace Blueberry
{
	class Object;

	class Variant
	{
	public:
		Variant() = default;

		Variant(void* data);
		Variant(void* data, const uint32_t& offset);

		template<class ObjectType>
		operator const ObjectType*();

		template<class ObjectType>
		ObjectType* Get();

		void* Get() { return m_Data; }
		
	private:
		void* m_Data;
	};

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

	