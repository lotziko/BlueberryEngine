#include "EntityHelper.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"

namespace Blueberry
{
	Entity* EntityHelper::Create(const String& name)
	{
		Entity* entity = Object::Create<Entity>();
		entity->SetName(name);
		entity->AddComponent<Transform>();
		return entity;
	}

	void EntityHelper::GatherObjects(Entity* entity, List<Object*>& objects)
	{
		objects.push_back(entity);
		for (size_t i = 0; i < entity->GetComponentCount(); i++)
		{
			objects.push_back(entity->GetComponentAt(i));
		}
		for (auto& child : entity->GetTransform()->GetChildren())
		{
			GatherObjects(child.Get()->GetEntity(), objects);
		}
	}
}