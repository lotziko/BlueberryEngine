#include "bbpch.h"
#include "Object.h"

namespace Blueberry
{
	const std::size_t Object::Type = std::hash<std::string>()(TO_STRING(Object));
	const std::size_t Object::ParentType = 0;

	std::map<ObjectId, Ref<Object>> ObjectDB::s_Objects = std::map<ObjectId, Ref<Object>>();
	ObjectId ObjectDB::s_Id = 0;

	void Object::Serialize(ryml::NodeRef& node)
	{
	}

	void Object::Deserialize(ryml::NodeRef& node)
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

	Guid Object::GetGuid() const
	{
		return m_Guid;
	}

	void ObjectDB::DestroyObject(Ref<Object>& object)
	{
		s_Objects.erase(object->GetObjectId());
	}

	void ObjectDB::DestroyObject(Object* object)
	{
		s_Objects.erase(object->GetObjectId());
	}
}