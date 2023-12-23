#pragma once

#include "Object.h"
#include "ObjectDB.h"

namespace Blueberry
{
	template<class ObjectType>
	class ObjectPtr
	{
	public:
		ObjectPtr() = default;
		ObjectPtr(const ObjectType* object);
		void operator=(const ObjectType* object);
		bool operator==(const ObjectPtr<ObjectType>& other) const;
		ObjectType* operator->();

		ObjectType* Get() const;
		bool IsValid() const;
		void Reset();

	private:
		ObjectId m_Id = INVALID_ID;
		ObjectItem* m_Item = nullptr;
	};

	template<class ObjectType>
	inline ObjectPtr<ObjectType>::ObjectPtr(const ObjectType* object)
	{
		if (object != nullptr)
		{
			m_Id = object->GetObjectId();
			m_Item = ObjectDB::IdToObjectItem(m_Id);
		}
		else
		{
			Reset();
		}
	}

	template<class ObjectType>
	inline void ObjectPtr<ObjectType>::operator=(const ObjectType* object)
	{
		if (object != nullptr)
		{
			m_Id = object->GetObjectId();
			m_Item = ObjectDB::IdToObjectItem(m_Id);
		}
		else
		{
			Reset();
		}
	}

	template<class ObjectType>
	inline bool ObjectPtr<ObjectType>::operator==(const ObjectPtr<ObjectType>& other) const
	{
		return m_Id == other.m_Id || !IsValid() && !other.IsValid();
	}

	template<class ObjectType>
	inline ObjectType* ObjectPtr<ObjectType>::operator->()
	{
		return ObjectPtr<ObjectType>::Get();
	}

	template<class ObjectType>
	inline ObjectType* ObjectPtr<ObjectType>::Get() const
	{
		if (m_Id < 0)
		{
			return nullptr;
		}
		return m_Item != nullptr ? (ObjectType*)m_Item->object : nullptr;
	}

	template<class ObjectType>
	inline bool ObjectPtr<ObjectType>::IsValid() const
	{
		return m_Item != nullptr && m_Item->object != nullptr;
	}

	template<class ObjectType>
	inline void ObjectPtr<ObjectType>::Reset()
	{
		m_Id = INVALID_ID;
		m_Item = nullptr;
	}
}