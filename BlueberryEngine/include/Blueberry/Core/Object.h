#pragma once

#include "Base.h"
#include "Memory.h"

namespace Blueberry
{
//********************************************************************************
// OBJECT_DECLARATION
// This macro must be included in the declaration of any subclass of Object.
// It declares variables used in type checking.
//********************************************************************************
#define OBJECT_DECLARATION( classname )											\
public:																			\
    static TypeId Type;															\
    static TypeId ParentType;													\
	static const String TypeName;												\
	static const String ParentTypeName;											\
	static void DefineFields();													\
public:																			\
    virtual bool IsClassType(TypeId classType) const override;					\
	virtual TypeId GetType() const override;									\
	virtual const String& GetTypeName() const override;							\

//********************************************************************************
// OBJECT_DEFINITION
// This macro must be included in the class definition to properly initialize 
// variables used in type checking. Take special care to ensure that the 
// proper parentclass is indicated or the run-time type information will be incorrect.
//********************************************************************************
#define OBJECT_DEFINITION( childclass, parentclass )								\
TypeId childclass::Type = 0;														\
TypeId childclass::ParentType = 0;													\
const String childclass::TypeName = TO_STRING(childclass);							\
const String childclass::ParentTypeName = TO_STRING(parentclass);					\
bool childclass::IsClassType(TypeId classType) const								\
{																					\
    if ( classType == childclass::Type )											\
        return true;																\
    return parentclass::IsClassType(classType);										\
}																					\
TypeId childclass::GetType() const													\
{																					\
	return childclass::Type;														\
}																					\
const String& childclass::GetTypeName() const										\
{																					\
	return childclass::TypeName;													\
}																					\
void childclass::DefineFields()														\

	enum class ObjectState
	{
		AwaitingLoading,
		Loading,
		Default,
		Missing
	};

	class BB_API Object
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		static TypeId Type;
		static TypeId ParentType;
		static const String TypeName;
		static const String ParentTypeName;

	public:
		Object() = default;
		virtual ~Object() = default;

		virtual bool IsClassType(TypeId classType) const;
		virtual TypeId GetType() const;
		virtual const String& GetTypeName() const;

	public:
		ObjectId GetObjectId() const;

		virtual const String& GetName();
		void SetName(const String& name);

		const ObjectState& GetState();
		const bool IsDefaultState();
		void SetState(ObjectState state);

		static void DefineFields();

		template<class ObjectType>
		static ObjectType* Create();

		static Object* Clone(Object* object);
		static void Destroy(Object* object);

	protected:
		ObjectId m_ObjectId = INVALID_ID;
		String m_Name;
		ObjectState m_State = ObjectState::Default;

		friend class ObjectDB;
		friend class ClassDB;
		friend struct ClassInfo;
		friend class Serializer;
	};

#define DATA_DECLARATION( classname )											\
public:																			\
    static TypeId Type;															\
	static const String TypeName;												\
	static void DefineFields();													\

#define DATA_DEFINITION( classname )											\
TypeId classname::Type = 0;										\
const String classname::TypeName = TO_STRING(classname);						\
void classname::DefineFields()													\

	class Data
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		Data() = default;
		~Data() = default;

		static const TypeId Type;
		static const String TypeName;
	};

	template<class ObjectType>
	inline ObjectType* Object::Create()
	{
		ObjectType* object = new ObjectType();
		ObjectDB::AllocateId(object);
		return object;
	}
}