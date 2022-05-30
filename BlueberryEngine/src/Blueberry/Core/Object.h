#pragma once

#include <string>

//********************************************************************************
// OBJECT_DECLARATION
// This macro must be included in the declaration of any subclass of Object.
// It declares variables used in type checking.
//********************************************************************************
#define OBJECT_DECLARATION( classname )										\
public:                                                                     \
    static const std::size_t Type;											\
public:																		\
    virtual bool IsClassType( const std::size_t classType ) const override;	\
	virtual std::size_t GetType() const override;							\

//********************************************************************************
// OBJECT_DEFINITION
// This macro must be included in the class definition to properly initialize 
// variables used in type checking. Take special care to ensure that the 
// proper parentclass is indicated or the run-time type information will be incorrect.
//********************************************************************************
#define OBJECT_DEFINITION( parentclass, childclass )											\
const std::size_t childclass::Type = std::hash< std::string >()( TO_STRING( childclass ) );		\
bool childclass::IsClassType( const std::size_t classType ) const								\
{																								\
    if ( classType == childclass::Type )														\
        return true;																			\
    return parentclass::IsClassType( classType );												\
}																								\
std::size_t childclass::GetType() const															\
{																								\
	return childclass::Type;																	\
}																								\

class Object
{
public:
	virtual bool IsClassType(const std::size_t classType) const
	{
		return classType == Type;
	}
	virtual std::size_t GetType() const
	{
		return Type;
	}
	virtual std::string ToString() const
	{
		return "Object";
	}
public:
	static const std::size_t Type;
};