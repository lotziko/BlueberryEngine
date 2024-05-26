#pragma once

#include <string>
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\MethodBind.h"
#include "Blueberry\Core\FieldBind.h"
#include "Blueberry\Tools\StringHelper.h"

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

		IntByteArray,
		FloatByteArray,

		Enum,

		Vector2,
		Vector3,
		Vector4,
		Quaternion,

		Color,

		// Derived from Object
		ObjectPtr,
		ObjectPtrArray,

		// Not derived from Object
		Data,
		DataArray
	};

	struct FieldInfo
	{
		std::string name;
		FieldBind* bind;
		MethodBind* setter;
		BindingType type;
		std::size_t objectType;
		void* hintData;

		template<class ObjectType, class FieldType>
		FieldInfo(const std::string& name, FieldType ObjectType::* field, const BindingType& type);

		FieldInfo SetHintData(char* hintData);
		FieldInfo SetObjectType(const std::size_t& objectType);
		template<class ObjectType, class FieldType>
		FieldInfo SetSetter(void(ObjectType::* setter)(FieldType));
	};

	class ClassDB
	{
	public:

		struct BindingData
		{
			std::vector<FieldInfo> fieldInfos;

			BindingData& BindField(FieldInfo info)
			{
				fieldInfos.emplace_back(std::move(info));
				return *this;
			}
		};

		struct ClassInfo
		{
			std::string name;
			std::size_t parentId;
			Object*(*createInstance)() = nullptr;
			Data*(*createDataInstance)() = nullptr;
			bool isObject;
			size_t offset;
			std::vector<FieldInfo> fields;
			std::unordered_map<std::string, FieldInfo> fieldsMap;
		};

		static const ClassInfo& GetInfo(const std::size_t&);
		static std::unordered_map<std::size_t, ClassInfo>& GetInfos();
		static bool IsParent(const std::size_t& id, const std::size_t& parentId);

		template<class ObjectType>
		static void Register();
		template<class ObjectType>
		static void RegisterAbstract();
		template <class ObjectType>
		static void RegisterData();
		template<class ObjectType>
		static void Bind(BindingData bindings);

	private:
		template<class ObjectType>
		static Object* CreateObject()
		{
			return Object::Create<ObjectType>();
		}

		template<class ObjectType>
		static Data* CreateData()
		{
			return new ObjectType();
		}

	private:
		static std::unordered_map<std::size_t, ClassInfo> s_Classes;
	};

	constexpr auto GetFieldName(std::string_view name)
	{
		return name.substr(name.find_last_of(':') + 1);
	}
	
	#define REGISTER_CLASS( classname ) ClassDB::Register<classname>();
	#define REGISTER_ABSTRACT_CLASS( classname ) ClassDB::RegisterAbstract<classname>();
	#define REGISTER_DATA_CLASS( classname ) ClassDB::RegisterData<classname>();

	#define BEGIN_OBJECT_BINDING( classname ) ClassDB::Bind<classname>(ClassDB::BindingData()
	#define BIND_FIELD( fieldInfo ) .BindField(fieldInfo)
	#define END_OBJECT_BINDING() );

	template<class ObjectType>
	inline void ClassDB::Register()
	{
		std::size_t id = ObjectType::Type;
		std::size_t parentId = ObjectType::ParentType;
		std::string name = ObjectType::TypeName;
		Object*(*createFunction)() = &ClassDB::CreateObject<ObjectType>;
		size_t offset = reinterpret_cast<char*>(static_cast<Object*>(reinterpret_cast<ObjectType*>(0x10000000))) - reinterpret_cast<char*>(0x10000000);

		if (s_Classes.count(id) == 0)
		{
			s_Classes.insert({ id, { name, parentId, createFunction, nullptr, true, offset } });
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
			s_Classes.insert({ id, { name, parentId, nullptr, nullptr, true, 0 } });
		}

		ObjectType::BindProperties();
	}

	template<class ObjectType>
	inline void ClassDB::RegisterData()
	{
		std::size_t id = ObjectType::Type;
		std::size_t parentId = 0;
		std::string name = ObjectType::TypeName;
		Data*(*createFunction)() = &ClassDB::CreateData<ObjectType>;
		size_t offset = reinterpret_cast<char*>(static_cast<Data*>(reinterpret_cast<ObjectType*>(0x10000000))) - reinterpret_cast<char*>(0x10000000);

		if (s_Classes.count(id) == 0)
		{
			s_Classes.insert({ id, { name, parentId, nullptr, createFunction, false, offset } });
		}

		ObjectType::BindProperties();
	}

	template<class ObjectType>
	inline void ClassDB::Bind(BindingData bindings)
	{
		auto classInfoIt = s_Classes.find(ObjectType::Type);
		if (classInfoIt != s_Classes.end())
		{
			for (FieldInfo info : bindings.fieldInfos)
			{
				classInfoIt->second.fields.emplace_back(info);
				classInfoIt->second.fieldsMap.insert_or_assign(info.name, info);
			}
		}
	}

	template<class ObjectType, class FieldType>
	inline FieldInfo::FieldInfo(const std::string& name, FieldType ObjectType::* field, const BindingType& type) : name(name), bind(FieldBind::Create(reinterpret_cast<FieldType ObjectType::*>(field))), type(type), objectType(0), hintData(nullptr)
	{
	}

	template<class ObjectType, class FieldType>
	inline FieldInfo FieldInfo::SetSetter(void(ObjectType::* setter)(FieldType))
	{
		this->setter = MethodBind::Create(reinterpret_cast<void(void::*)(FieldType)>(setter));
		return *this;
	}
}