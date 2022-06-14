#include "bbpch.h"
#include "EntityInspector.h"

#include "Blueberry\Scene\Entity.h"

#include "imgui\imgui.h"

namespace Blueberry
{
	OBJECT_INSPECTOR_DECLARATION(EntityInspector, Entity)

	void EntityInspector::Draw(Object* object)
	{
		Entity* entity = static_cast<Entity*>(object);
		std::vector<Ref<Component>> components = entity->GetComponents();

		ImGui::Text(entity->ToString().c_str());

		for (auto component : components)
		{
			std::size_t type = component->GetType();
			ObjectInspector* inspector = ObjectInspectors::GetInspector(type);

			if (inspector != nullptr)
			{
				inspector->Draw(component.get());
			}
		}
	}
}