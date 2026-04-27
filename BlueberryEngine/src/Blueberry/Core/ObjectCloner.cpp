#include "Blueberry\Core\ObjectCloner.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Core\Variant.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Component.h"
#include "Blueberry\Scene\Components\Transform.h"

namespace Blueberry
{
	void GatherHierarchy(Entity* entity, List<Object*>& objects)
	{
		List<Entity*> entitiesToClone;
		entitiesToClone.push_back(entity);
		while (entitiesToClone.size() > 0)
		{
			entity = entitiesToClone.back();
			entitiesToClone.pop_back();
			objects.push_back(entity);
			for (size_t i = 0; i < entity->GetComponentCount(); ++i)
			{
				Component* component = entity->GetComponentAt(i);
				if (component != nullptr)
				{
					objects.push_back(component);
				}
			}
			for (auto& child : entity->GetTransform()->GetChildren())
			{
				Entity* childEntity = child->GetEntity();
				if (childEntity != nullptr)
				{
					entitiesToClone.push_back(childEntity);
				}
			}
		}
	}

	Object* ObjectCloner::Clone(Object* object)
	{
		ObjectMapping mapping;
		List<Object*> objectsToClone;
		Entity* rootEntity = nullptr;
		if (object->IsClassType(Component::Type))
		{
			GatherHierarchy(rootEntity = static_cast<Component*>(object)->GetEntity(), objectsToClone);
		}
		else if (object->IsClassType(Entity::Type))
		{
			GatherHierarchy(rootEntity = static_cast<Entity*>(object), objectsToClone);
		}
		else
		{
			objectsToClone.push_back(object);
		}
		List<std::tuple<Object*, Object*, const ClassInfo*>> objectsToCopy;
		for (Object* object : objectsToClone)
		{
			const ClassInfo* info = ClassDB::GetInfo(object->GetType());
			Object* clone = info->Create();
			mapping.insert_or_assign(object->GetObjectId(), clone->GetObjectId());
			objectsToCopy.push_back(std::make_tuple(object, clone, info));
		}
		for (auto& tuple : objectsToCopy)
		{
			CopyFields(mapping, std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple));
		}
		if (rootEntity != nullptr)
		{
			Entity* newRootEntity = static_cast<Entity*>(ObjectDB::GetObject(mapping[rootEntity->GetObjectId()]));
			ClassDB::GetInfo(Transform::Type)->GetField("m_Parent")->Set(newRootEntity->GetTransform(), ObjectPtr<Transform>());
		}
		return ObjectDB::GetObject(mapping[object->GetObjectId()]);
	}

	Object* ObjectCloner::Resolve(ObjectMapping& mapping, List<ObjectId>& removed, Object* object)
	{
		HashSet<ObjectId> visited;
		List<Object*> objectsToClone;
		if (object->IsClassType(Component::Type))
		{
			GatherHierarchy(static_cast<Component*>(object)->GetEntity(), objectsToClone);
		}
		else if (object->IsClassType(Entity::Type))
		{
			GatherHierarchy(static_cast<Entity*>(object), objectsToClone);
		}
		else
		{
			objectsToClone.push_back(object);
		}
		List<std::tuple<Object*, Object*, const ClassInfo*>> objectsToCopy;
		for (Object* object : objectsToClone)
		{
			const ClassInfo* info = ClassDB::GetInfo(object->GetType());
			Object* clone = nullptr;
			auto it = mapping.find(object->GetObjectId());
			if (it != mapping.end())
			{
				clone = ObjectDB::GetObject(it->second);
			}
			else
			{
				clone = info->Create();
				mapping.insert_or_assign(object->GetObjectId(), clone->GetObjectId());
			}
			visited.insert(object->GetObjectId());
			objectsToCopy.push_back(std::make_tuple(object, clone, info));
		}
		for (auto& tuple : objectsToCopy)
		{
			CopyFields(mapping, std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple));
		}
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
		return ObjectDB::GetObject(mapping[object->GetObjectId()]);
	}

	void ObjectCloner::CopyFields(ObjectMapping& mapping, void* source, void* target, const ClassInfo* info)
	{
		for (auto& field : info->fields)
		{
			switch (field.type)
			{
			case BindingType::Bool:
				*field.Get<bool>(target) = *field.Get<bool>(source);
				break;
			case BindingType::Int:
				*field.Get<int>(target) = *field.Get<int>(source);
				break;
			case BindingType::Uint:
				*field.Get<uint32_t>(target) = *field.Get<uint32_t>(source);
				break;
			case BindingType::Long:
				*field.Get<long>(target) = *field.Get<long>(source);
				break;
			case BindingType::Ulong:
				*field.Get<unsigned long>(target) = *field.Get<unsigned long>(source);
				break;
			case BindingType::Float:
				*field.Get<float>(target) = *field.Get<float>(source);
				break;
			case BindingType::String:
				*field.Get<String>(target) = *field.Get<String>(source);
				break;
			case BindingType::ByteData:
				*field.Get<ByteData>(target) = *field.Get<ByteData>(source);
				break;
			case BindingType::IntList:
				*field.Get<List<int>>(target) = *field.Get<List<int>>(source);
				break;
			case BindingType::FloatList:
				*field.Get<List<float>>(target) = *field.Get<List<float>>(source);
				break;
			case BindingType::StringList:
				*field.Get<List<String>>(target) = *field.Get<List<String>>(source);
				break;
			case BindingType::Enum:
				*field.Get<int>(target) = *field.Get<int>(source);
				break;
			case BindingType::Vector2:
				*field.Get<Vector2>(target) = *field.Get<Vector2>(source);
				break;
			case BindingType::Vector2Int:
				*field.Get<Vector2Int>(target) = *field.Get<Vector2Int>(source);
				break;
			case BindingType::Vector3:
				*field.Get<Vector3>(target) = *field.Get<Vector3>(source);
				break;
			case BindingType::Vector3Int:
				*field.Get<Vector3Int>(target) = *field.Get<Vector3Int>(source);
				break;
			case BindingType::Vector4:
				*field.Get<Vector4>(target) = *field.Get<Vector4>(source);
				break;
			case BindingType::Vector4Int:
				*field.Get<Vector4Int>(target) = *field.Get<Vector4Int>(source);
				break;
			case BindingType::Quaternion:
				*field.Get<Quaternion>(target) = *field.Get<Quaternion>(source);
				break;
			case BindingType::Color:
				*field.Get<Color>(target) = *field.Get<Color>(source);
				break;
			case BindingType::AABB:
				*field.Get<AABB>(target) = *field.Get<AABB>(source);
				break;
			case BindingType::Matrix:
				*field.Get<Matrix>(target) = *field.Get<Matrix>(source);
				break;
			case BindingType::Vector2List:
				*field.Get<List<Vector2>>(target) = *field.Get<List<Vector2>>(source);
				break;
			case BindingType::Vector3List:
				*field.Get<List<Vector3>>(target) = *field.Get<List<Vector3>>(source);
				break;
			case BindingType::Vector4List:
				*field.Get<List<Vector4>>(target) = *field.Get<List<Vector4>>(source);
				break;
			case BindingType::QuaternionList:
				*field.Get<List<Quaternion>>(target) = *field.Get<List<Quaternion>>(source);
				break;
			case BindingType::MatrixList:
				*field.Get<List<Matrix>>(target) = *field.Get<List<Matrix>>(source);
				break;
			case BindingType::Raw:
				memcpy(field.Get<uint8_t>(target), field.Get<uint8_t>(source), field.options.size);
				break;
			case BindingType::ObjectPtr:
			{
				ObjectPtr<Object> objectRefValue = *field.Get<ObjectPtr<Object>>(source);
				Object* object = nullptr;
				if (objectRefValue.IsValid())
				{
					object = objectRefValue.Get();
					auto it = mapping.find(object->GetObjectId());
					if (it != mapping.end())
					{
						object = ObjectDB::GetObject(it->second);
					}
				}
				*field.Get<ObjectPtr<Object>>(target) = object;
			}
			break;
			case BindingType::ObjectPtrList:
			{
				List<ObjectPtr<Object>>* objectRefArrayValue = field.Get<List<ObjectPtr<Object>>>(source);
				List<ObjectPtr<Object>>* cloneObjectRefArrayPointer = field.Get<List<ObjectPtr<Object>>>(target);

				cloneObjectRefArrayPointer->clear();
				if (objectRefArrayValue->size() > 0)
				{
					for (auto it = objectRefArrayValue->begin(); it != objectRefArrayValue->end(); ++it)
					{
						if (it->IsValid())
						{
							Object* object = it->Get();
							auto mappingIt = mapping.find(object->GetObjectId());
							if (mappingIt != mapping.end())
							{
								object = ObjectDB::GetObject(mappingIt->second);
							}
							cloneObjectRefArrayPointer->push_back(object);
						}
						else
						{
							cloneObjectRefArrayPointer->push_back(nullptr);
						}
					}
				}
			}
			break;
			case BindingType::Data:
			{
				Data* objectData = field.Get<Data>(source);
				Data* cloneData = field.Get<Data>(target);
				const ClassInfo* info = ClassDB::GetInfo(*field.options.objectType);
				if (info != nullptr)
				{
					CopyFields(mapping, objectData, cloneData, info);
				}
				else
				{
					BB_ERROR("Data class not exists.");
				}
			}
			break;
			case BindingType::DataList:
			{
				ListBase* objectDataArrayPointer = field.Get<ListBase>(source);
				ListBase* cloneDataArrayPointer = field.Get<ListBase>(target);
				const ClassInfo* info = ClassDB::GetInfo(*field.options.objectType);
				if (info != nullptr)
				{
					cloneDataArrayPointer->clear_base();
					for (size_t i = 0; i < objectDataArrayPointer->size_base(); ++i)
					{
						void* objectData = objectDataArrayPointer->get_base(i);
						void* cloneData = cloneDataArrayPointer->emplace_back_base();
						CopyFields(mapping, objectData, cloneData, info);
					}
				}
				else
				{
					BB_ERROR("Data class not exists.");
				}
			}
			break;
			case BindingType::Variant:
				*field.Get<Variant>(target) = *field.Get<Variant>(source);
				break;
			default:
				BB_INFO("Can't clone field " << field.name);
				break;
			}
		}
	}
}
