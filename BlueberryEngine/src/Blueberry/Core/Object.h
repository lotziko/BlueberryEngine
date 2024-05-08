#pragma once

#include <string>

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
    static const std::size_t Type;												\
    static const std::size_t ParentType;										\
	static const std::string TypeName;											\
public:																			\
    virtual bool IsClassType( const std::size_t classType ) const override;		\
	virtual std::size_t GetType() const override;								\
	virtual std::string GetTypeName() const override;							\
	static void PopulateParentTypes(std::set<size_t>& types);					\

//********************************************************************************
// OBJECT_DEFINITION
// This macro must be included in the class definition to properly initialize 
// variables used in type checking. Take special care to ensure that the 
// proper parentclass is indicated or the run-time type information will be incorrect.
//********************************************************************************
#define OBJECT_DEFINITION( parentclass, childclass )								\
const std::size_t childclass::Type = TO_OBJECT_TYPE(TO_STRING(childclass));			\
const std::size_t childclass::ParentType = TO_OBJECT_TYPE(TO_STRING(parentclass));	\
const std::string childclass::TypeName = TO_STRING(childclass);						\
bool childclass::IsClassType( const std::size_t classType ) const					\
{																					\
    if ( classType == childclass::Type )											\
        return true;																\
    return parentclass::IsClassType( classType );									\
}																					\
std::size_t childclass::GetType() const												\
{																					\
	return childclass::Type;														\
}																					\
std::string childclass::GetTypeName() const											\
{																					\
	return childclass::TypeName;													\
}																					\
void childclass::PopulateParentTypes(std::set<size_t>& types)						\
{																					\
	types.emplace(childclass::Type);												\
	parentclass::PopulateParentTypes(types);										\
}																					\

	using ObjectId = int32_t;

	enum class ObjectState
	{
		AwaitingLoading,
		Loading,
		Default,
		Missing
	};

	class Object
	{
	public:
		static const std::size_t Type;
		static const std::size_t ParentType;
		static const std::string TypeName;
		static std::set<size_t> ParentTypes;

	public:
		Object();
		~Object();

		virtual bool IsClassType(const std::size_t classType) const;
		virtual std::size_t GetType() const;
		virtual std::string GetTypeName() const;
		static void PopulateParentTypes(std::set<size_t>& types);

	public:
		ObjectId GetObjectId() const;

		const std::string& GetName();
		void SetName(const std::string& name);

		const ObjectState& GetState();
		const bool IsDefaultState();
		void SetState(const ObjectState& state);

		static void BindProperties();

		template<class ObjectType>
		static ObjectType* Create();

		static Object* Clone(Object* object);
		static void Destroy(Object* object);

		virtual void OnCreate() { };
		virtual void OnDestroy() { };

	protected:
		ObjectId m_ObjectId;
		std::string m_Name;
		ObjectState m_State = ObjectState::Default;

		friend class ObjectDB;
		friend class ClassDB;
		friend class Serializer;
	};

	template<class ObjectType>
	inline ObjectType* Object::Create()
	{
		ObjectType* object = new ObjectType();
		return object;
	}
}