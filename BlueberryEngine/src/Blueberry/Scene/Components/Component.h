#pragma once

#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class Entity;
	class Component;

	class ComponentDefinitions
	{
	public:
		struct Info
		{
			std::string name;
			std::function<Ref<Component>()> createInstance;
		};

		static std::map<std::size_t, Info>& GetDefinitions()
		{
			static std::map<std::size_t, Info> s_Definitions = std::map<std::size_t, Info>();
			return s_Definitions;
		}

		ComponentDefinitions(const std::size_t& id, const std::string& name, const std::function<Ref<Component>()>&& createFunction) 
		{ 
			ComponentDefinitions::GetDefinitions().insert({ id, { name, createFunction } }); 
		}
	};

//********************************************************************************
// COMPONENT_DECLARATION
// This macro must be included in the declaration of any subclass of Component.
// It declares method to create instances of the component.
//********************************************************************************
#define COMPONENT_DECLARATION( componentType )								\
public:																		\
	static Ref<Component> CreateInstance();									\

//********************************************************************************
// COMPONENT_DEFINITION
// This macro must be included in the class definition to properly initialize 
// method used in component instance creation. Take special care to ensure that the 
//********************************************************************************
#define COMPONENT_DEFINITION( componentType )																															\
	Ref<Component> componentType::CreateInstance() { return CreateRef<componentType>(); }																				\
	static ComponentDefinitions ComponentDefinition_##componentType( componentType::Type, #componentType, std::bind<Ref<Component>>(&componentType::CreateInstance) );	\

	class Component : public Object
	{
		OBJECT_DECLARATION(Component)
	public:
		virtual ~Component() = default;

		inline Entity* GetEntity() { return m_Entity; }

	private:
		Entity* m_Entity;

		friend class Entity;
	};
}