#pragma once

#include <string>
#include "Blueberry\Core\Guid.h"

namespace Blueberry
{

//********************************************************************************
// OBJECT_DECLARATION
// This macro must be included in the declaration of any subclass of Object.
// It declares variables used in type checking.
//********************************************************************************
#define OBJECT_DECLARATION( classname )										\
public:                                                                     \
    static const std::size_t Type;											\
    static const std::size_t ParentType;									\
	static const std::string TypeName;										\
public:																		\
    virtual bool IsClassType( const std::size_t classType ) const override;	\
	virtual std::size_t GetType() const override;							\
	virtual std::string GetTypeName() const override;						\

//********************************************************************************
// OBJECT_DEFINITION
// This macro must be included in the class definition to properly initialize 
// variables used in type checking. Take special care to ensure that the 
// proper parentclass is indicated or the run-time type information will be incorrect.
//********************************************************************************
#define OBJECT_DEFINITION( parentclass, childclass )												\
const std::size_t childclass::Type = std::hash< std::string >()( TO_STRING( childclass ) );			\
const std::size_t childclass::ParentType = std::hash< std::string >()( TO_STRING( parentclass ) );	\
const std::string childclass::TypeName = TO_STRING( childclass );									\
bool childclass::IsClassType( const std::size_t classType ) const									\
{																									\
    if ( classType == childclass::Type )															\
        return true;																				\
    return parentclass::IsClassType( classType );													\
}																									\
std::size_t childclass::GetType() const																\
{																									\
	return childclass::Type;																		\
}																									\
std::string childclass::GetTypeName() const															\
{																									\
	return childclass::TypeName;																	\
}																									\

	using ObjectId = uint64_t;

	class Object
	{
	public:
		static const std::size_t Type;
		static const std::size_t ParentType;
		static const std::string TypeName;

	public:
		virtual bool IsClassType(const std::size_t classType) const;
		virtual std::size_t GetType() const;
		virtual std::string GetTypeName() const;
		ObjectId GetObjectId() const;
		Guid& GetGuid() const;

		const std::string& GetName();
		void SetName(const std::string& name);

		static void BindProperties();

	protected:
		ObjectId m_ObjectId;
		std::string m_Name;

		friend class ObjectDB;
	};

	class ObjectDB
	{
	public:
		template<class ObjectType, typename... Args>
		static Ref<ObjectType> CreateObject(Args&&... params);
		template<class ObjectType, typename... Args>
		static Ref<ObjectType> CreateGuidObject(const Guid& guid, Args&&... params);
		static void DestroyObject(Ref<Object>& object);
		static void DestroyObject(Object* object);

	private:
		static std::map<ObjectId, Ref<Object>> s_Objects;
		static std::map<ObjectId, Guid> s_ObjectIdToGuid;
		static ObjectId s_MaxId;

		friend class Object;
	};

	template<class ObjectType, typename... Args>
	inline Ref<ObjectType> ObjectDB::CreateObject(Args&&... params)
	{
		static_assert(std::is_base_of<Object, ObjectType>::value, "Type is not derived from Object.");

		ObjectId id = ++s_MaxId;
		auto& object = CreateRef<ObjectType>(std::forward<Args>(params)...);
		object->m_ObjectId = id;
		s_Objects.insert({ id, object });

		return object;
	}

	template<class ObjectType, typename... Args>
	inline Ref<ObjectType> ObjectDB::CreateGuidObject(const Guid& guid, Args&&... params)
	{
		static_assert(std::is_base_of<Object, ObjectType>::value, "Type is not derived from Object.");

		ObjectId id = ++s_MaxId;
		Ref<ObjectType> object = CreateObject<ObjectType>(std::forward<Args>(params)...);
		object->m_ObjectId = id;
		s_Objects.insert({ id, object });
		s_ObjectIdToGuid.insert({ id, guid });

		return object;
	}
}