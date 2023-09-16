#pragma once

#include <string>

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
public:																		\
    virtual bool IsClassType( const std::size_t classType ) const override;	\
	virtual std::size_t GetType() const override;							\

//********************************************************************************
// OBJECT_DEFINITION
// This macro must be included in the class definition to properly initialize 
// variables used in type checking. Take special care to ensure that the 
// proper parentclass is indicated or the run-time type information will be incorrect.
//********************************************************************************
#define OBJECT_DEFINITION( parentclass, childclass )												\
const std::size_t childclass::Type = std::hash< std::string >()( TO_STRING( childclass ) );			\
const std::size_t childclass::ParentType = std::hash< std::string >()( TO_STRING( parentclass ) );	\
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

	class Object
	{
	public:
		static const std::size_t Type;
		static const std::size_t ParentType;
	public:
		virtual bool IsClassType(const std::size_t classType) const;
		virtual std::size_t GetType() const;
		virtual std::string ToString() const;
	};

	using ObjectId = uint64_t;

	class ObjectDB
	{
	private:
		static ObjectId AddInstance(Object* object);
		static void RemoveInstance(Object* object);

	private:
		static std::vector<Object> s_Objects;

		friend class Object;
	};
}