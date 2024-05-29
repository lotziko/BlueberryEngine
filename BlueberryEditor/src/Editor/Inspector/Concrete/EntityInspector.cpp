#include "bbpch.h"
#include "EntityInspector.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Component.h"

#include "imgui\imgui.h"
#include "Blueberry\Core\ClassDB.h"

#include "Editor\Inspector\ObjectInspectorDB.h"

namespace Blueberry
{
	bool ShouldHideType(const std::size_t type, const ClassDB::ClassInfo& info)
	{
		if (info.createInstance == nullptr)
		{
			return true;
		}

		if (!ClassDB::IsParent(type, Component::Type))
		{
			return true;
		}

		return false;
	}

	void EntityInspector::Draw(Object* object)
	{
		Entity* entity = static_cast<Entity*>(object);
		std::vector<Component*> components = entity->GetComponents();

		ImGui::Text(entity->GetTypeName().c_str());
		
		// Name
	/*	{
			ImGui::Text("Name");
			ImGui::SameLine();

			static ObjectId entityId;
			static char buf[256];
			if (entity->GetObjectId() != entityId)
			{
				std::string name = entity->GetName();
				strncpy(buf, name.c_str(), sizeof(buf) - 1);
				entityId = entity->GetObjectId();
			}

			ImGui::InputText("##name", buf, 256);
			if (ImGui::IsItemDeactivated())
			{
				entity->SetName(buf);
			}
		}*/

		for (auto component : components)
		{
			std::size_t type = component->GetType();
			std::string name = component->GetTypeName();
			ObjectInspector* inspector = ObjectInspectorDB::GetInspector(type);

			if (inspector != nullptr)
			{
				const char* headerId = name.c_str();
				const char* popupId = "InspectorPopup";

				ImGui::PushID(headerId);

				if (ImGui::CollapsingHeader(headerId))
				{
					inspector->Draw(component);
				}

				if (ImGui::BeginPopup(popupId))
				{
					if (ImGui::MenuItem("Remove component"))
					{
						entity->RemoveComponent(component);
					}
					ImGui::EndPopup();
				}

				ImGui::OpenPopupOnItemClick(popupId, ImGuiPopupFlags_MouseButtonRight);

				ImGui::PopID();
			}
		}

		if (ImGui::Button("Add component..."))
		{
			ImGui::OpenPopup("addComponent");
		}
		
		if (ImGui::BeginPopup("addComponent"))
		{
			auto infos = ClassDB::GetInfos();
			
			for (auto info : infos)
			{
				if (ShouldHideType(info.first, info.second))
				{
					continue;
				}

				if (ImGui::Selectable(info.second.name.c_str()))
				{
					entity->AddComponent((Component*)info.second.createInstance());
				}
			}

			ImGui::EndPopup();
		}
	}
}