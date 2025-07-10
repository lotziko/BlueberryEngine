#include "Blueberry\Core\ObjectCloner.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Core\Variant.h"
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
		Object* clone = info.createInstance();
		mapping.insert_or_assign(object->GetObjectId(), clone->GetObjectId());

		Variant originalValue;
		Variant cloneValue;
		for (auto field : info.fields)
		{
			originalValue = Variant(object, field.offset);
			cloneValue = Variant(clone, field.offset);

			switch (field.type)
			{
			case BindingType::Bool:
				*cloneValue.Get<bool>() = *originalValue.Get<bool>();
				break;
			case BindingType::Int:
				*cloneValue.Get<int>() = *originalValue.Get<int>();
				break;
			case BindingType::Float:
				*cloneValue.Get<float>() = *originalValue.Get<float>();
				break;
			case BindingType::String:
				*cloneValue.Get<String>() = *originalValue.Get<String>();
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
				List<ObjectPtr<Object>> objectRefArrayValue = *originalValue.Get<List<ObjectPtr<Object>>>();
				List<ObjectPtr<Object>>* cloneObjectRefArrayPointer = cloneValue.Get<List<ObjectPtr<Object>>>();
				
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
						else
						{
							cloneObjectRefArrayPointer->emplace_back(nullptr);
						}
					}
				}
			}
			break;
			case BindingType::Enum:
				*cloneValue.Get<int>() = *originalValue.Get<int>();
				break;
			case BindingType::Vector2:
				*cloneValue.Get<Vector2>() = *originalValue.Get<Vector2>();
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
		clone->OnCreate();
		return clone;
	}
}
