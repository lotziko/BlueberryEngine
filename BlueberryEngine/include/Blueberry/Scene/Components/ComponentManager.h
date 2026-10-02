#pragma once

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	class ComponentBucket
	{
	private:
		List<Component*> m_Components;
		Dictionary<ObjectId, size_t> m_Positions;

		template<class ComponentType>
		friend class ComponentIterator;
		template<class ComponentType>
		friend class ComponentView;
		friend class ComponentManager;
	};

	template<class ComponentType>
	class ComponentIterator
	{
	public:
		ComponentIterator(ComponentBucket* bucket, size_t index) : m_Bucket(bucket), m_Index(index)
		{
		}

		ComponentIterator<ComponentType>& operator++();
		ComponentIterator<ComponentType>& operator--();

		ComponentType* operator*() const;
		ComponentType* operator->() const;

		bool operator!= (ComponentIterator<ComponentType> other) const;
		bool operator== (ComponentIterator<ComponentType> other) const;

	private:
		ComponentBucket* m_Bucket;
		size_t m_Index;

		template<class ComponentType>
		friend class ComponentView;
	};

	template<class ComponentType>
	class ComponentView
	{
	public:
		ComponentView(ComponentBucket* bucket) : m_Bucket(bucket) {}

		ComponentIterator<ComponentType> begin() const;
		ComponentIterator<ComponentType> end() const;

	private:
		ComponentBucket* m_Bucket;
	};

	class ComponentManager
	{
	public:
		void AddComponent(Component* component);
		// Unsafe
		void AddComponent(Component* component, TypeId type);
		void RemoveComponent(Component* component);
		// Unsafe
		void RemoveComponent(Component* component, TypeId type);

		template<class ComponentType>
		ComponentView<ComponentType> GetComponents();
		ComponentView<Component> GetComponents(TypeId type);

	private:
		Dictionary<TypeId, ComponentBucket> m_Buckets;
	};

	template<class ComponentType>
	inline ComponentIterator<ComponentType>& ComponentIterator<ComponentType>::operator++()
	{
		m_Index += 1;
		return *this;
	}

	template<class ComponentType>
	inline ComponentIterator<ComponentType>& ComponentIterator<ComponentType>::operator--()
	{
		m_Index -= 1;
		return *this;
	}

	template<class ComponentType>
	inline ComponentType* ComponentIterator<ComponentType>::operator*() const
	{
		return static_cast<ComponentType*>(m_Bucket->m_Components[m_Index]);
	}

	template<class ComponentType>
	inline ComponentType* ComponentIterator<ComponentType>::operator->() const
	{
		return static_cast<ComponentType*>(m_Bucket->m_Components[m_Index]);
	}

	template<class ComponentType>
	inline bool ComponentIterator<ComponentType>::operator!=(ComponentIterator<ComponentType> other) const
	{
		return m_Index != other.m_Index;
	}

	template<class ComponentType>
	inline bool ComponentIterator<ComponentType>::operator==(ComponentIterator<ComponentType> other) const
	{
		return m_Index == other.m_Index;
	}

	template<class ComponentType>
	inline ComponentIterator<ComponentType> ComponentView<ComponentType>::begin() const
	{
		return ComponentIterator<ComponentType>(m_Bucket, 0);
	}

	template<class ComponentType>
	inline ComponentIterator<ComponentType> ComponentView<ComponentType>::end() const
	{
		return ComponentIterator<ComponentType>(m_Bucket, m_Bucket->m_Components.size());
	}

	inline void ComponentManager::AddComponent(Component* component)
	{
		AddComponent(component, component->GetType());
	}

	inline void ComponentManager::AddComponent(Component* component, TypeId type)
	{
		ComponentBucket& bucket = m_Buckets[type];
		ObjectId id = component->GetObjectId();
		if (bucket.m_Positions.count(id) == 0)
		{
			bucket.m_Positions.insert_or_assign(id, bucket.m_Components.size());
			bucket.m_Components.push_back(component);
		}
	}

	inline void ComponentManager::RemoveComponent(Component* component)
	{
		RemoveComponent(component, component->GetType());
	}

	inline void ComponentManager::RemoveComponent(Component* component, TypeId type)
	{
		ComponentBucket& bucket = m_Buckets[type];
		ObjectId id = component->GetObjectId();
		auto it = bucket.m_Positions.find(id);
		if (it != bucket.m_Positions.end())
		{
			size_t position = it->second;
			if (position != bucket.m_Components.size() - 1)
			{
				Component* movedComponent = bucket.m_Components.back();
				bucket.m_Components[position] = movedComponent;
				bucket.m_Positions[movedComponent->GetObjectId()] = position;
			}
			bucket.m_Components.pop_back();
			bucket.m_Positions.erase(it);
		}
	}

	template<class ComponentType>
	inline ComponentView<ComponentType> ComponentManager::GetComponents()
	{
		return ComponentView<ComponentType>(&m_Buckets[ComponentType::Type]);
	}

	inline ComponentView<Component> ComponentManager::GetComponents(TypeId type)
	{
		return ComponentView<Component>(&m_Buckets[type]);
	}
}