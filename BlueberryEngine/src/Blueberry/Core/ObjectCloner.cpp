#include "bbpch.h"
#include "ObjectCloner.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Component.h"
#include "Blueberry\Scene\Components\Transform.h"

namespace Blueberry
{
	Object* ObjectCloner::Clone(Object* object)
	{
		ObjectMapping objectMapping;
		Object* clone = CloneObject(objectMapping, object);
		return clone;
	}

	Object* ObjectCloner::CloneObject(ObjectMapping& mapping, Object* object)
	{
		// TODO iterate object fields and clone them too if they have references to objects inside

		ClassDB::ClassInfo info = ClassDB::GetInfo(object->GetType());
		Object* clone = (Object*)info.createInstance();
		mapping.insert_or_assign(object->GetObjectId(), clone->GetObjectId());

		Variant originalValue;
		Variant cloneValue;
		for (auto field : info.fields)
		{
			field.bind->Get(object, originalValue);
			field.bind->Get(clone, cloneValue);

			switch (field.type)
			{
			case BindingType::String:
				*cloneValue.Get<std::string>() = *originalValue.Get<std::string>();
				break;
			case BindingType::ObjectPtr:
			{
				ObjectPtr<Object> objectRefValue = *originalValue.Get<ObjectPtr<Object>>();
				if (objectRefValue.IsValid())
				{
					Object* child = objectRefValue.Get();
					if (!ObjectDB::HasGuid(child) || child->IsClassType(Component::Type) || child->IsClassType(Entity::Type))
					{
						ObjectId childId = child->GetObjectId();
						auto mappingIt = mapping.find(childId);
						if (mappingIt != mapping.end())
						{
							child = ObjectDB::GetObject(mappingIt->second);
						}
						else
						{
							child = CloneObject(mapping, child);
						}
					}
					*cloneValue.Get<ObjectPtr<Object>>() = child;
				}
			}
			break;
			case BindingType::ObjectPtrArray:
			{
				std::vector<ObjectPtr<Object>> objectRefArrayValue = *originalValue.Get<std::vector<ObjectPtr<Object>>>();
				std::vector<ObjectPtr<Object>>* cloneObjectRefArrayPointer = cloneValue.Get<std::vector<ObjectPtr<Object>>>();
				
				if (objectRefArrayValue.size() > 0)
				{
					for (ObjectPtr<Object>& objectRefValue : objectRefArrayValue)
					{
						if (objectRefValue.IsValid())
						{
							Object* child = objectRefValue.Get();
							if (!ObjectDB::HasGuid(child) || child->IsClassType(Component::Type) || child->IsClassType(Entity::Type))
							{
								ObjectId childId = child->GetObjectId();
								auto mappingIt = mapping.find(childId);
								if (mappingIt != mapping.end())
								{
									child = ObjectDB::GetObject(mappingIt->second);
								}
								else
								{
									child = CloneObject(mapping, child);
								}
							}
							cloneObjectRefArrayPointer->emplace_back(child);
						}
					}
				}
			}
			break;
			case BindingType::Vector3:
				*cloneValue.Get<Vector3>() = *originalValue.Get<Vector3>();
				break;
			case BindingType::Vector4:
				*cloneValue.Get<Vector4>() = *originalValue.Get<Vector4>();
				break;
			case BindingType::Quaternion:
				*cloneValue.Get<Quaternion>() = *originalValue.Get<Quaternion>();
				break;
			case BindingType::Color:
				*cloneValue.Get<Color>() = *originalValue.Get<Color>();
				break;
			default:
				BB_INFO("Can't clone field " << field.name);
				break;
			}
		}
		return clone;
	}
}