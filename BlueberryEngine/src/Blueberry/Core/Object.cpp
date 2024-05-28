#include "bbpch.h"
#include "Object.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Core\ObjectCloner.h"

namespace Blueberry
{
	const std::size_t Object::Type = TO_OBJECT_TYPE(TO_STRING(Object));
	const std::size_t Object::ParentType = 0;
	const std::string Object::TypeName = "Object";
	
	Object::Object()
	{
		ObjectDB::AllocateId(this);
	}

	Object::~Object()
	{
		ObjectDB::FreeId(this);
	}

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

	const std::string& Object::GetName()
	{
		return m_Name;
	}

	void Object::SetName(const std::string& name)
	{
		m_Name = name;
	}

	const ObjectState& Object::GetState()
	{
		return m_State;
	}

	const bool Object::IsDefaultState()
	{
		return m_State == ObjectState::Default;
	}

	void Object::SetState(const ObjectState& state)
	{
		m_State = state;
	}

	void Object::BindProperties()
	{
		BEGIN_OBJECT_BINDING(Object)
		BIND_FIELD(FieldInfo(TO_STRING(m_Name), &Object::m_Name, BindingType::String))
		END_OBJECT_BINDING()
	}

	Object* Object::Clone(Object* object)
	{
		return ObjectCloner::Clone(object);
	}

	void Object::Destroy(Object* object)
	{
		delete object;
	}
}