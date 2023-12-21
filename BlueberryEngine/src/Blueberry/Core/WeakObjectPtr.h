#pragma once

#include "Object.h"
#include "ObjectDB.h"

namespace Blueberry
{
	template<class ObjectType>
	class WeakObjectPtr
	{
	public:
		WeakObjectPtr() = default;
		WeakObjectPtr(const ObjectType* object);
		void operator=(const ObjectType* object);
		bool operator==(const WeakObjectPtr<ObjectType>& other) const;
		ObjectType* operator->();

		ObjectType* Get() const;
		bool IsValid() const;
		void Reset();
	private:
		ObjectId m_Id = INVALID_ID;
	};

	template<class ObjectType>
	inline WeakObjectPtr<ObjectType>::WeakObjectPtr(const ObjectType* object)
	{
		if (object != nullptr)
		{
			m_Id = object->GetObjectId();
		}
		else
		{
			Reset();
		}
	}

	template<class ObjectType>
	inline void WeakObjectPtr<ObjectType>::operator=(const ObjectType* object)
	{
		if (object != nullptr)
		{
			m_Id = object->GetObjectId();
		}
		else
		{
			Reset();
		}
	}

	template<class ObjectType>
	inline bool WeakObjectPtr<ObjectType>::operator==(const WeakObjectPtr<ObjectType>& other) const
	{
		return m_Id == other.m_Id || !IsValid() && !other.IsValid();
	}

	template<class ObjectType>
	inline ObjectType* WeakObjectPtr<ObjectType>::operator->()
	{
		return WeakObjectPtr<ObjectType>::Get();
	}

	template<class ObjectType>
	inline ObjectType* WeakObjectPtr<ObjectType>::Get() const
	{
		if (m_Id < 0)
		{
			return nullptr;
		}
		ObjectItem* item = ObjectDB::IdToObjectItem(m_Id);
		return item != nullptr ? (ObjectType*)item->object : nullptr;
	}

	template<class ObjectType>
	inline bool WeakObjectPtr<ObjectType>::IsValid() const
	{
		ObjectItem* item = ObjectDB::IdToObjectItem(m_Id);
		return item != nullptr && item->object != nullptr;
	}

	template<class ObjectType>
	inline void WeakObjectPtr<ObjectType>::Reset()
	{
		m_Id = INVALID_ID;
	}
}