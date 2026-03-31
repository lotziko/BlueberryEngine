#include "Blueberry\Core\Object.h"

#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Core\ObjectCloner.h"

namespace Blueberry
{
	TypeId Object::Type = 0;
	TypeId Object::ParentType = 0;
	const String Object::TypeName = "Object";
	const String Object::ParentTypeName = "";
	
	bool Object::IsClassType(const TypeId classType) const
	{
		return classType == Type;
	}

	TypeId Object::GetType() const
	{
		return Type;
	}

	const String& Object::GetTypeName() const
	{
		return TypeName;
	}

	ObjectId Object::GetObjectId() const
	{
		return m_ObjectId;
	}

	const String& Object::GetName()
	{
		return m_Name;
	}

	void Object::SetName(const String& name)
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

	void Object::SetState(ObjectState state)
	{
		m_State = state;
	}

	void Object::DefineFields()
	{
		DEFINE_FIELD(Object, m_Name, BindingType::String, FieldOptions().SetVisibility(VisibilityType::Hidden))
	}

	Object* Object::Clone(Object* object)
	{
		return ObjectCloner::Clone(object);
	}

	void Object::Destroy(Object* object)
	{
		ObjectDB::FreeId(object);
		delete object;
	}
}