#pragma once

#include <string>
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\MethodBind.h"
#include "Blueberry\Core\FieldBind.h"

namespace Blueberry
{
	enum class BindingType
	{
		None = 0,

		Bool,
		Int,
		Float,
		String,
		ByteData,

		Vector2,
		Vector3,
		Vector4,
		Quaternion,

		Color,

		Object,
		ObjectRef,
		ObjectPointerArray,
		ObjectRefArray
	};

	class ClassDB
	{
	public:
		struct FieldInfo
		{
			std::string name;
			FieldBind* bind;
			BindingType type;
		};

		struct PropertyInfo
		{
			std::string name;
			MethodBind* getter;
			MethodBind* setter;
			BindingType type;
		};

		template<class ObjectType>
		struct BindingData
		{
			std::vector<FieldInfo> fieldInfos;
			std::vector<PropertyInfo> propertyInfos;

			template<typename Field>
			BindingData& BindField(const std::string& name, Field field, const BindingType& type);

			template<typename Getter, typename Setter>
			BindingData& BindProperty(const std::string& name, Getter getter, Setter setter, const BindingType& type);
		};

		struct ClassInfo
		{
			std::string name;
			std::size_t parentId;
			Ref<Object>(*createInstance)() = nullptr;
			std::vector<FieldInfo> fields;
			std::vector<PropertyInfo> properties;
		};

		static const ClassInfo& GetInfo(const std::size_t&);
		static std::map<std::size_t, ClassInfo>& GetInfos();
		static bool IsParent(const std::size_t& id, const std::size_t& parentId);

		template<class ObjectType>
		static void Register();
		template<class ObjectType>
		static void RegisterAbstract();
		template<class ObjectType>
		static void Bind(BindingData<ObjectType> bindings);

	private:
		template<class ObjectType>
		static Ref<Object> Create()
		{
			return ObjectDB::CreateObject<ObjectType>();
		}

	private:
		static std::map<std::size_t, ClassInfo> s_Classes;
	};

	#define REGISTER_CLASS( classname ) ClassDB::Register<classname>();
	#define REGISTER_ABSTRACT_CLASS( classname ) ClassDB::RegisterAbstract<classname>();

	#define BEGIN_OBJECT_BINDING( classname ) ClassDB::Bind(ClassDB::BindingData<classname>()
	#define BIND_FIELD( name, field, type ) .BindField<decltype(field)>(name, field, type)
	#define BIND_PROPERTY( name, getter, setter, type ) .BindProperty<decltype(getter), decltype(setter)>(name, getter, setter, type)
	#define END_OBJECT_BINDING() );

	template<class ObjectType>
	inline void ClassDB::Register()
	{
		std::size_t id = ObjectType::Type;
		std::size_t parentId = ObjectType::ParentType;
		std::string name = ObjectType::TypeName;
		Ref<Object>(*createFunction)() = &ClassDB::Create<ObjectType>;

		if (s_Classes.count(id) == 0)
		{
			s_Classes.insert({ id, { name, parentId, createFunction } });
		}

		ObjectType::BindProperties();
	}

	template<class ObjectType>
	inline void ClassDB::RegisterAbstract()
	{
		std::size_t id = ObjectType::Type;
		std::size_t parentId = ObjectType::ParentType;
		std::string name = ObjectType::TypeName;

		if (s_Classes.count(id) == 0)
		{
			s_Classes.insert({ id, { name, parentId, nullptr } });
		}

		ObjectType::BindProperties();
	}

	template<class ObjectType>
	inline void ClassDB::Bind(BindingData<ObjectType> bindings)
	{
		auto classInfoIt = s_Classes.find(ObjectType::Type);
		if (classInfoIt != s_Classes.end())
		{
			for (FieldInfo info : bindings.fieldInfos)
			{
				classInfoIt->second.fields.emplace_back(info);
			}

			for (PropertyInfo info : bindings.propertyInfos)
			{
				classInfoIt->second.properties.emplace_back(info);
			}
		}
	}

	template<class ObjectType>
	template<typename Field>
	inline ClassDB::BindingData<ObjectType>& ClassDB::BindingData<ObjectType>::BindField(const std::string& name, Field field, const BindingType& type)
	{
		FieldInfo info = { name, FieldBind::Create<ObjectType>(field), type };
		fieldInfos.emplace_back(std::move(info));
		return *this;
	}

	template<class ObjectType>
	template<typename Getter, typename Setter>
	inline ClassDB::BindingData<ObjectType>& ClassDB::BindingData<ObjectType>::BindProperty(const std::string& name, Getter getter, Setter setter, const BindingType& type)
	{
		PropertyInfo info = { name, MethodBind::Create<ObjectType>(getter), MethodBind::Create<ObjectType>(setter), type };
		propertyInfos.emplace_back(std::move(info));
		return *this;
	}
}