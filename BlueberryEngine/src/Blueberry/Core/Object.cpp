#include "bbpch.h"
#include "Object.h"

#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	const std::size_t Object::Type = std::hash<std::string>()(TO_STRING(Object));
	const std::size_t Object::ParentType = 0;
	const std::string Object::TypeName = "Object";

	std::map<ObjectId, Ref<Object>> ObjectDB::s_Objects = std::map<ObjectId, Ref<Object>>();
	std::map<ObjectId, Guid> ObjectDB::s_ObjectIdToGuid = std::map<ObjectId, Guid>();
	ObjectId ObjectDB::s_MaxId = 0;
	
	bool Object::IsClassType(const std::size_t classType) const
	{
		return classType == Type;
	}

	std::size_t Object::GetType() const
	{
		return Type;
	}

	std::string Object::GetTypeName() const
	{
		return TypeName;
	}

	ObjectId Object::GetObjectId() const
	{
		return m_ObjectId;
	}

	Guid& Object::GetGuid() const
	{
		return ObjectDB::s_ObjectIdToGuid.find(m_ObjectId)->second;
	}

	const std::string& Object::GetName()
	{
		return m_Name;
	}

	void Object::SetName(const std::string& name)
	{
		m_Name = name;
	}

	void Object::BindProperties()
	{
		BEGIN_OBJECT_BINDING(Object)
		BIND_FIELD("m_Name", &Object::m_Name, BindingType::String)
		END_OBJECT_BINDING()
	}

	void ObjectDB::AddObjectGuid(const ObjectId& id, const Guid& guid)
	{
		s_ObjectIdToGuid.insert({ id, guid });
	}

	void ObjectDB::DestroyObject(Ref<Object>& object)
	{
		ObjectId id = object->GetObjectId();
		s_Objects.erase(id);
		s_ObjectIdToGuid.erase(id);
	}

	void ObjectDB::DestroyObject(Object* object)
	{
		ObjectId id = object->GetObjectId();
		s_Objects.erase(id);
		s_ObjectIdToGuid.erase(id);
	}
}