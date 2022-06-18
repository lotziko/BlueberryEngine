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
			std::string name = component->ToString();
			ObjectInspector* inspector = ObjectInspectors::GetInspector(type);

			if (inspector != nullptr)
			{
				const char* headerId = name.c_str();
				const char* popupId = (name + "_popup").c_str();

				ImGui::PushID(headerId);

				if (ImGui::CollapsingHeader(headerId))
				{
					inspector->Draw(component.get());
				}

				if (ImGui::IsItemHovered() && ImGui::IsMouseDown(1))
					ImGui::OpenPopup(popupId);

				if (ImGui::BeginPopup(popupId))
				{
					if (ImGui::MenuItem("Remove component"))
					{
						entity->RemoveComponent(component);
					}
					ImGui::EndPopup();
				}

				ImGui::PopID();
			}
		}

		if (ImGui::Button("Add component..."))
		{
			ImGui::OpenPopup("addComponent");
		}
		
		if (ImGui::BeginPopup("addComponent"))
		{
			auto definitions = ComponentDefinitions::GetDefinitions();
			
			for (auto definition : definitions)
			{
				if (ImGui::Selectable(definition.second.name.c_str()))
				{
					entity->AddComponent(definition.second.createInstance());
				}
			}

			ImGui::EndPopup();
		}
	}
}