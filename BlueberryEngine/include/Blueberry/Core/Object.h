#pragma once

#include "Base.h"
#include "Memory.h"

namespace Blueberry
{
#define TO_OBJECT_TYPE( classname ) TO_HASH( classname )

	//********************************************************************************
	// OBJECT_DECLARATION
	// This macro must be included in the declaration of any subclass of Object.
	// It declares variables used in type checking.
	//********************************************************************************
#define OBJECT_DECLARATION( classname )											\
public:																			\
    static const size_t Type;													\
    static const size_t ParentType;												\
	static const String TypeName;												\
	static void DefineFields();													\
public:																			\
    virtual bool IsClassType( const size_t classType ) const override;			\
	virtual size_t GetType() const override;									\
	virtual String GetTypeName() const override;								\

//********************************************************************************
// OBJECT_DEFINITION
// This macro must be included in the class definition to properly initialize 
// variables used in type checking. Take special care to ensure that the 
// proper parentclass is indicated or the run-time type information will be incorrect.
//********************************************************************************
#define OBJECT_DEFINITION( childclass, parentclass )								\
const size_t childclass::Type = TO_OBJECT_TYPE(TO_STRING(childclass));				\
const size_t childclass::ParentType = TO_OBJECT_TYPE(TO_STRING(parentclass));		\
const String childclass::TypeName = TO_STRING(childclass);							\
bool childclass::IsClassType( const size_t classType ) const						\
{																					\
    if ( classType == childclass::Type )											\
        return true;																\
    return parentclass::IsClassType( classType );									\
}																					\
size_t childclass::GetType() const													\
{																					\
	return childclass::Type;														\
}																					\
String childclass::GetTypeName() const												\
{																					\
	return childclass::TypeName;													\
}																					\
void childclass::DefineFields()														\

	using ObjectId = int32_t;

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

		static const size_t Type;
		static const size_t ParentType;
		static const String TypeName;

	public:
		Object();
		virtual ~Object();

		virtual bool IsClassType(const size_t classType) const;
		virtual size_t GetType() const;
		virtual String GetTypeName() const;

	public:
		ObjectId GetObjectId() const;

		const String& GetName();
		void SetName(const String& name);

		const ObjectState& GetState();
		const bool IsDefaultState();
		void SetState(const ObjectState& state);

		static void DefineFields();

		template<class ObjectType>
		static ObjectType* Create();

		static Object* Clone(Object* object);
		static void Destroy(Object* object);

		virtual void OnCreate() { };
		virtual void OnDestroy() { };

	protected:
		ObjectId m_ObjectId;
		String m_Name;
		ObjectState m_State = ObjectState::Default;

		friend class ObjectDB;
		friend class ClassDB;
		friend class Serializer;
	};

#define DATA_DECLARATION( classname )											\
public:																			\
    static const size_t Type;													\
	static const String TypeName;												\
	static void DefineFields();													\

#define DATA_DEFINITION( classname )											\
const size_t classname::Type = TO_OBJECT_TYPE(TO_STRING(classname));			\
const String classname::TypeName = TO_STRING(classname);						\
void classname::DefineFields()													\

	class Data
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		Data() = default;
		~Data() = default;

		static const size_t Type;
		static const String TypeName;
	};

	template<class ObjectType>
	inline ObjectType* Object::Create()
	{
		ObjectType* object = new ObjectType();
		return object;
	}
}