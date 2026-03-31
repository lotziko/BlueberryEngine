#include "Blueberry\Scene\SceneEvents.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	List<ObjectPtr<Component>> SceneEvents::s_CreatedComponents = {};
	List<ObjectPtr<Component>> SceneEvents::s_EnabledComponents = {};
	List<ObjectPtr<Component>> SceneEvents::s_DisabledComponents = {};
	List<ObjectPtr<Component>> SceneEvents::s_DestroyedComponents = {};
	List<ObjectPtr<Entity>> SceneEvents::s_DestroyedEntities = {};

	void SceneEvents::Poll()
	{
		if (s_CreatedComponents.size() > 0)
		{
			for (auto& createdComponent : s_CreatedComponents)
			{
				Component* component = createdComponent.Get();
				if (component != nullptr)
				{
					component->OnCreate();
				}
			}
			s_CreatedComponents.clear();
		}
		if (s_EnabledComponents.size() > 0)
		{
			for (auto& enabledComponent : s_EnabledComponents)
			{
				Component* component = enabledComponent.Get();
				if (component != nullptr)
				{
					component->OnEnable();
				}
			}
			s_EnabledComponents.clear();
		}
		if (s_DisabledComponents.size() > 0)
		{
			for (auto& disabledComponent : s_DisabledComponents)
			{
				Component* component = disabledComponent.Get();
				if (component != nullptr)
				{
					component->OnDisable();
				}
			}
			s_DisabledComponents.clear();
		}
		if (s_DestroyedComponents.size() > 0)
		{
			for (auto& destroyedComponent : s_DestroyedComponents)
			{
				Component* component = destroyedComponent.Get();
				if (component != nullptr)
				{
					component->OnDestroy();
					Object::Destroy(component);
				}
			}
			s_DestroyedComponents.clear();
		}
		if (s_DestroyedEntities.size() > 0)
		{
			for (auto& destroyedEntity : s_DestroyedEntities)
			{
				Entity* entity = destroyedEntity.Get();
				if (entity != nullptr)
				{
					Object::Destroy(entity);
				}
			}
			s_DestroyedEntities.clear();
		}
	}
}