#pragma once

#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Component;
	class Transform;
	class Scene;

	class BB_API Entity : public Object
	{
		OBJECT_DECLARATION(Entity)
		
	public:
		Entity() = default;
		Entity(const String& name);

		void OnCreate();
		void OnDestroy();

		template<class ComponentType>
		ComponentType* AddComponent();

		void AddComponent(Component* component);

		template<class ComponentType>
		ComponentType* GetComponent();

		Component* GetComponentAt(size_t index) const;

		template<class ComponentType>
		ComponentType* GetComponentInParent();
		template<class ComponentType>
		ComponentType* GetComponentInChildren();
		template<class ComponentType>
		List<ComponentType*> GetComponentsInChildren();

		const size_t GetComponentCount();

		template<class ComponentType>
		bool HasComponent();

		void RemoveComponent(Component* component);

		Transform* GetTransform();
		Scene* GetScene() const;
		
		bool IsActive() const;
		void SetActive(bool active);
		bool IsActiveInHierarchy();

		void UpdateHierarchy();

	private:
		bool HasComponent(TypeId type);
		Component* GetComponent(TypeId type);
		Component* GetComponentInParent(TypeId type);
		Component* GetComponentInChildren(TypeId type);
		List<Component*> GetComponentsInChildren(TypeId type);
		void AddComponentToScene(Component* component);
		void RemoveComponentFromScene(Component* component);

	private:
		void UpdateHierarchy(bool active);
		void UpdateComponents();
		void EnableComponents();
		void DisableComponents();

	private:
		List<ObjectPtr<Component>> m_Components;
		bool m_IsActive = true;

		Transform* m_Transform = nullptr;
		Scene* m_Scene = nullptr;
		bool m_IsActiveInHierarchy = false;

		friend class Scene;
		friend class Component;
		friend class Transform;
	};

	template<class ComponentType>
	inline ComponentType* Entity::AddComponent()
	{
		static_assert(std::is_base_of<Component, ComponentType>::value, "Type is not derived from Component.");
		ComponentType* component = Object::Create<ComponentType>();
		AddComponent(component);
		return component;
	}

	template<class ComponentType>
	inline ComponentType* Entity::GetComponent()
	{
		static_assert(std::is_base_of<Component, ComponentType>::value, "Type is not derived from Component.");
		return static_cast<ComponentType*>(GetComponent(ComponentType::Type));
	}

	template<class ComponentType>
	inline ComponentType* Entity::GetComponentInParent()
	{
		static_assert(std::is_base_of<Component, ComponentType>::value, "Type is not derived from Component.");
		return static_cast<ComponentType*>(GetComponentInParent(ComponentType::Type));
	}

	template<class ComponentType>
	inline ComponentType* Entity::GetComponentInChildren()
	{
		static_assert(std::is_base_of<Component, ComponentType>::value, "Type is not derived from Component.");
		return static_cast<ComponentType*>(GetComponentInChildren(ComponentType::Type));
	}

	template<class ComponentType>
	inline List<ComponentType*> Entity::GetComponentsInChildren()
	{
		static_assert(std::is_base_of<Component, ComponentType>::value, "Type is not derived from Component.");
		List<Component*> components = GetComponentsInChildren(ComponentType::Type);
		List<ComponentType*> result(components.size());
		for (size_t i = 0; i < components.size(); ++i)
		{
			result[i] = static_cast<ComponentType*>(components[i]);
		}
		return result;
	}

	template<class ComponentType>
	inline bool Entity::HasComponent()
	{
		static_assert(std::is_base_of<Component, ComponentType>::value, "Type is not derived from Component.");
		return HasComponent(ComponentType::Type);
	}
}