#include "Blueberry\Scene\Entity.h"

#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Scene\Scene.h"

namespace Blueberry
{
	std::priority_queue<Entity::OperationData, List<Entity::OperationData>, Entity::CompareOperations> Entity::s_Operations;

	OBJECT_DEFINITION(Entity, Object)
	{
		DEFINE_BASE_FIELDS(Entity, Object)
		DEFINE_FIELD(Entity, m_Components, BindingType::ObjectPtrList, FieldOptions().SetObjectType(&Component::Type).SetVisibility(VisibilityType::Hidden))
		DEFINE_FIELD(Entity, m_IsActive, BindingType::Bool, FieldOptions().SetUpdateCallback(MethodBind::Create(&Entity::UpdateHierarchy)))
	}

	Entity::Entity(const String& name)
	{
		SetName(name);
	}

	void Entity::OnCreate()
	{
	}

	void Entity::OnDestroy()
	{
		if (!m_IsDestroyed)
		{
			for (auto& componentSlot : m_Components)
			{
				if (componentSlot.IsValid())
				{
					Component* component = componentSlot.Get();
					if (component->CanExecute())
					{
						if (component->m_IsActive)
						{
							s_Operations.push({ 3000, Operation::DisableComponent, component });
							component->m_IsActive = false;
						}
						if (!component->m_IsDestroyed)
						{
							s_Operations.push({ 4000, Operation::DestroyComponent, component });
							component->m_IsDestroyed = true;
						}
					}
				}
			}
			m_Components.clear();
			s_Operations.push({ 5000, Operation::DestroyEntity, this });
			m_IsDestroyed = true;
		}
	}

	void Entity::AddComponent(Component* component)
	{
		int index = 0;
		for (auto& componentSlot : m_Components)
		{
			if (!componentSlot.IsValid())
			{
				componentSlot = component;
				break;
			}
			++index;
		}

		component->m_Entity = this;
		if (index >= m_Components.size())
		{
			m_Components.push_back(component);
		}
		if (IsActiveInHierarchy())
		{
			if (component->CanExecute())
			{
				if (!component->m_IsCreated)
				{
					s_Operations.push({ 1000, Operation::CreateComponent, component });
					component->m_IsCreated = true;
				}
				if (!component->m_IsActive)
				{
					s_Operations.push({ 2000, Operation::EnableComponent, component });
					component->m_IsActive = true;
				}
			}
		}
	}

	Component* Entity::GetComponentAt(size_t index) const
	{
		if (index >= 0 && index < m_Components.size())
		{
			return m_Components[index].Get();
		}
		return nullptr;
	}

	const size_t Entity::GetComponentCount()
	{
		return m_Components.size();
	}

	void Entity::RemoveComponent(Component* component)
	{
		if (component->m_Entity == this)
		{
			if (component->CanExecute())
			{
				if (component->m_IsActive)
				{
					s_Operations.push({ 3000, Operation::DisableComponent, component });
					component->m_IsActive = false;
				}
				if (!component->m_IsDestroyed)
				{
					s_Operations.push({ 4000, Operation::DestroyComponent, component });
					component->m_IsDestroyed = true;
				}
			}
			auto& index = std::find(m_Components.begin(), m_Components.end(), component);
			m_Components.erase(index);
		}
	}

	Transform* Entity::GetTransform()
	{
		if (m_Transform == nullptr)
		{
			m_Transform = static_cast<Transform*>(m_Components[0].Get());
		}
		return m_Transform;
	}

	Scene* Entity::GetScene() const
	{
		return m_Scene;
	}

	bool Entity::IsActive() const
	{
		return m_IsActive;
	}

	void Entity::SetActive(bool active)
	{
		if (active != m_IsActive)
		{
			m_IsActive = active;
			UpdateHierarchy(active);
		}
	}

	bool Entity::IsActiveInHierarchy()
	{
		return m_IsActiveInHierarchy;
	}

	void Entity::UpdateHierarchy()
	{
		if (m_Scene == nullptr)
		{
			return;
		}
		Transform* parent = GetTransform()->GetParent();
		if (parent != nullptr)
		{
			UpdateHierarchy(parent->GetEntity()->m_IsActiveInHierarchy);
		}
		else
		{
			UpdateHierarchy(true);
		}
	}

	void Entity::Poll()
	{
		if (s_Operations.size() > 0)
		{
			while (!s_Operations.empty())
			{
				OperationData data = s_Operations.top();
				s_Operations.pop();
				Object* object = data.object.Get();
				if (object != nullptr)
				{
					switch (data.operation)
					{
					case Operation::CreateComponent:
						(static_cast<Component*>(object))->OnCreate();
						break;
					case Operation::EnableComponent:
					{
						Component* component = static_cast<Component*>(object);
						Scene* scene = component->GetEntity()->GetScene();
						if (scene != nullptr)
						{
							for (TypeId* type : ClassDB::GetInfo(component->GetType())->iterators)
							{
								scene->m_ComponentManager.AddComponent(component, *type);
							}
						}
						component->OnEnable();
					}
					break;
					case Operation::DisableComponent:
					{
						Component* component = static_cast<Component*>(object);
						Scene* scene = component->GetEntity()->GetScene();
						if (scene != nullptr)
						{
							for (TypeId* type : ClassDB::GetInfo(component->GetType())->iterators)
							{
								scene->m_ComponentManager.RemoveComponent(component, *type);
							}
						}
						component->OnDisable();
					}
					break;
					case Operation::DestroyComponent:
						(static_cast<Component*>(object))->OnDestroy();
						Object::Destroy(object);
						break;
					case Operation::DestroyEntity:
						Object::Destroy(object);
						break;
					}
				}
			}
		}
	}

	bool Entity::HasComponent(TypeId type)
	{
		for (auto& component : m_Components)
		{
			if (component.IsValid() && component->IsClassType(type))
			{
				return true;
			}
		}
		return false;
	}

	Component* Entity::GetComponent(TypeId type)
	{
		for (auto& component : m_Components)
		{
			if (component.IsValid() && component->IsClassType(type))
			{
				return component.Get();
			}
		}
		return nullptr;
	}

	Component* Entity::GetComponentInParent(TypeId type)
	{
		Entity* entity = this;
		while (entity != nullptr)
		{
			for (auto& component : entity->m_Components)
			{
				if (component.IsValid() && component->IsClassType(type))
				{
					return component.Get();
				}
			}
			Transform* parent = entity->GetTransform()->GetParent();
			if (parent == nullptr)
			{
				break;
			}
			entity = parent->GetEntity();
		}
		return nullptr;
	}

	Component* Entity::GetComponentInChildren(TypeId type)
	{
		List<Entity*> stack;
		stack.push_back(this);
		while (stack.size() > 0)
		{
			Entity* entity = stack[stack.size() - 1];
			Component* component = entity->GetComponent(type);
			if (component != nullptr)
			{
				return component;
			}
			stack.pop_back();
			for (auto& child : entity->GetTransform()->GetChildren())
			{
				stack.push_back(child.Get()->GetEntity());
			}
		}
		return nullptr;
	}

	List<Component*> Entity::GetComponentsInChildren(TypeId type)
	{
		List<Component*> result;
		List<Entity*> stack;
		stack.push_back(this);
		while (stack.size() > 0)
		{
			Entity* entity = stack[stack.size() - 1];
			for (size_t i = 0; i < entity->GetComponentCount(); ++i)
			{
				Component* component = entity->GetComponentAt(i);
				if (component->IsClassType(type))
				{
					result.push_back(component);
				}
			}
			stack.pop_back();
			for (auto& child : entity->GetTransform()->GetChildren())
			{
				stack.push_back(child.Get()->GetEntity());
			}
		}
		return result;
	}

	void Entity::UpdateHierarchy(bool active)
	{
		bool newActive = m_IsActive && active;
		if (m_IsActiveInHierarchy != newActive)
		{
			m_IsActiveInHierarchy = newActive;
			UpdateComponents();
		}
		if (GetTransform()->GetChildrenCount() > 0)
		{
			for (auto& child : GetTransform()->GetChildren())
			{
				child.Get()->GetEntity()->UpdateHierarchy(newActive);
			}
		}
	}

	void Entity::UpdateComponents()
	{
		if (m_IsActiveInHierarchy)
		{
			EnableComponents();
		}
		else
		{
			DisableComponents();
		}
	}

	void Entity::EnableComponents()
	{
		for (auto&& componentSlot : m_Components)
		{
			Component* component = componentSlot.Get();
			if (component->CanExecute())
			{
				if (!component->m_IsCreated)
				{
					s_Operations.push({ 1000, Operation::CreateComponent, component });
					component->m_IsCreated = true;
				}
				if (!component->m_IsActive)
				{
					s_Operations.push({ 2000, Operation::EnableComponent, component });
					component->m_IsActive = true;
				}
			}
		}
	}

	void Entity::DisableComponents()
	{
		for (auto&& componentSlot : m_Components)
		{
			Component* component = componentSlot.Get();
			if (component->m_IsActive && component->CanExecute())
			{
				s_Operations.push({ 3000, Operation::DisableComponent, component });
				component->m_IsActive = false;
			}
		}
	}
}