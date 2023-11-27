#include "bbpch.h"
#include "Object.h"

namespace Blueberry
{
	const std::size_t Object::Type = std::hash<std::string>()(TO_STRING(Object));
	const std::size_t Object::ParentType = 0;

	std::map<ObjectId, Ref<Object>> ObjectDB::s_Objects = std::map<ObjectId, Ref<Object>>();
	std::map<ObjectId, Guid> ObjectDB::s_ObjectIdToGuid = std::map<ObjectId, Guid>();
	ObjectId ObjectDB::s_MaxId = 0;

	void Object::Serialize(SerializationContext& context, ryml::NodeRef& node)
	{
	}

	void Object::Deserialize(SerializationContext& context, ryml::NodeRef& node)
	{
	}

	bool Object::IsClassType(const std::size_t classType) const
	{
		return classType == Type;
	}

	std::size_t Object::GetType() const
	{
		return Type;
	}

	std::string Object::ToString() const
	{
		return "Object";
	}

	ObjectId Object::GetObjectId() const
	{
		return m_ObjectId;
	}

	Guid& Object::GetGuid() const
	{
		return ObjectDB::s_ObjectIdToGuid.find(m_ObjectId)->second;
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