#include "Blueberry\Core\ObjectCloner.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Component.h"
#include "Blueberry\Scene\Components\Transform.h"

namespace Blueberry
{
	// Need something like clone or resolve for prefabs
	Object* ObjectCloner::Clone(Object* object)
	{
		HashSet<ObjectId> visited;
		ObjectMapping objectMapping;
		return CloneObject(visited, objectMapping, object);
	}

	Object* ObjectCloner::Clone(ObjectMapping& mapping, Object* object)
	{
		HashSet<ObjectId> visited;
		return CloneObject(visited, mapping, object);
	}

	Object* ObjectCloner::Resolve(ObjectMapping& mapping, List<ObjectId>& removed, Object* object)
	{
		HashSet<ObjectId> visited;
		Object* result = CloneObject(visited, mapping, object);
		List<ObjectId> removedIds;
		for (auto& pair : mapping)
		{
			if (visited.count(pair.first) == 0)
			{
				removed.push_back(pair.first);
			}
		}
		for (ObjectId id : removed)
		{
			mapping.erase(id);
		}
		return result;
	}

	Object* ObjectCloner::CloneObject(HashSet<ObjectId>& visited, ObjectMapping& mapping, Object* object)
	{
		// TODO iterate object fields and clone them too if they have references to objects inside

		const ClassInfo* info = ClassDB::GetInfo(object->GetType());
		if (info == nullptr)
		{
			BB_ERROR("Class not exists.");
			return nullptr;
		}
		Object* clone;

		auto mappingIt = mapping.find(object->GetObjectId());
		if (mappingIt != mapping.end())
		{
			clone = ObjectDB::GetObject(mappingIt->second);
			if (visited.count(object->GetObjectId()) > 0)
			{
				return clone;
			}
		}
		else
		{
			clone = info->createInstance();
			mapping.insert_or_assign(object->GetObjectId(), clone->GetObjectId());
		}
		visited.insert(object->GetObjectId());

		for (auto field : info->fields)
		{
			switch (field.type)
			{
			case BindingType::Bool:
				*field.Get<bool>(clone) = *field.Get<bool>(object);
				break;
			case BindingType::Int:
				*field.Get<int>(clone) = *field.Get<int>(object);
				break;
			case BindingType::Float:
				*field.Get<float>(clone) = *field.Get<float>(object);
				break;
			case BindingType::String:
				*field.Get<String>(clone) = *field.Get<String>(object);
				break;
			case BindingType::ObjectPtr:
			{
				ObjectPtr<Object> objectRefValue = *field.Get<ObjectPtr<Object>>(object);
				if (objectRefValue.IsValid())
				{
					Object* child = objectRefValue.Get();
					if (!ObjectDB::HasGuid(child) || child->IsClassType(Component::Type) || child->IsClassType(Entity::Type))
					{
						child = CloneObject(visited, mapping, child);
					}
					*field.Get<ObjectPtr<Object>>(clone) = child;
				}
			}
			break;
			case BindingType::ObjectPtrList:
			{
				List<ObjectPtr<Object>> objectRefArrayValue = *field.Get<List<ObjectPtr<Object>>>(object);
				List<ObjectPtr<Object>>* cloneObjectRefArrayPointer = field.Get<List<ObjectPtr<Object>>>(clone);
				
				if (objectRefArrayValue.size() > 0)
				{
					cloneObjectRefArrayPointer->clear();
					for (ObjectPtr<Object>& objectRefValue : objectRefArrayValue)
					{
						if (objectRefValue.IsValid())
						{
							Object* child = objectRefValue.Get();
							if (!ObjectDB::HasGuid(child) || child->IsClassType(Component::Type) || child->IsClassType(Entity::Type))
							{
								child = CloneObject(visited, mapping, child);
							}
							cloneObjectRefArrayPointer->push_back(child);
						}
						else
						{
							cloneObjectRefArrayPointer->push_back(nullptr);
						}
					}
				}
			}
			break;
			case BindingType::Enum:
				*field.Get<int>(clone) = *field.Get<int>(object);
				break;
			case BindingType::Vector2:
				*field.Get<Vector2>(clone) = *field.Get<Vector2>(object);
				break;
			case BindingType::Vector3:
				*field.Get<Vector3>(clone) = *field.Get<Vector3>(object);
				break;
			case BindingType::Vector4:
				*field.Get<Vector4>(clone) = *field.Get<Vector4>(object);
				break;
			case BindingType::Quaternion:
				*field.Get<Quaternion>(clone) = *field.Get<Quaternion>(object);
				break;
			case BindingType::Color:
				*field.Get<Color>(clone) = *field.Get<Color>(object);
				break;
			default:
				BB_INFO("Can't clone field " << field.name);
				break;
			}

			MethodBind* callback = field.options.updateCallback;
			if (callback != nullptr)
			{
				callback->Invoke(clone);
			}
		}
		clone->OnCreate();
		return clone;
	}
}
