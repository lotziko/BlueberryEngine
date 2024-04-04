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
		Ptr,
		PtrArray
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

		/*template<class ObjectType, class FieldType>
		FieldInfo(const std::string& name, FieldType ObjectType::* field, const BindingType& type, char* hintData);

		template<class ObjectType, class FieldType>
		FieldInfo(const std::string& name, FieldType ObjectType::* field, const BindingType& type, const std::size_t& objectType);
	
		template<class ObjectType, class FieldType>
		FieldInfo(const std::string& name, FieldType ObjectType::* field, void(ObjectType::* setter)(FieldType), const BindingType& type, const std::size_t& objectType);*/
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
			std::vector<FieldInfo> fields;
		};

		static const ClassInfo& GetInfo(const std::size_t&);
		static std::map<std::size_t, ClassInfo>& GetInfos();
		static bool IsParent(const std::size_t& id, const std::size_t& parentId);

		template<class ObjectType>
		static void Register();
		template<class ObjectType>
		static void RegisterAbstract();
		template<class ObjectType>
		static void Bind(BindingData bindings);

	private:
		template<class ObjectType>
		static Object* Create()
		{
			return Object::Create<ObjectType>();
		}

	private:
		static std::map<std::size_t, ClassInfo> s_Classes;
	};

	constexpr auto GetFieldName(std::string_view name)
	{
		return name.substr(name.find_last_of(':') + 1);
	}
	
	#define REGISTER_CLASS( classname ) ClassDB::Register<classname>();
	#define REGISTER_ABSTRACT_CLASS( classname ) ClassDB::RegisterAbstract<classname>();

	#define BEGIN_OBJECT_BINDING( classname ) ClassDB::Bind<classname>(ClassDB::BindingData()
	#define BIND_FIELD( fieldInfo ) .BindField(fieldInfo)
	#define END_OBJECT_BINDING() );

	template<class ObjectType>
	inline void ClassDB::Register()
	{
		std::size_t id = ObjectType::Type;
		std::size_t parentId = ObjectType::ParentType;
		std::string name = ObjectType::TypeName;
		Object*(*createFunction)() = &ClassDB::Create<ObjectType>;

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
	inline void ClassDB::Bind(BindingData bindings)
	{
		auto classInfoIt = s_Classes.find(ObjectType::Type);
		if (classInfoIt != s_Classes.end())
		{
			for (FieldInfo info : bindings.fieldInfos)
			{
				classInfoIt->second.fields.emplace_back(info);
			}
		}
	}

	template<class ObjectType, class FieldType>
	inline FieldInfo::FieldInfo(const std::string& name, FieldType ObjectType::* field, const BindingType& type) : name(name), bind(FieldBind::Create(reinterpret_cast<FieldType Object::*>(field))), type(type), objectType(0), hintData(nullptr)
	{
	}

	template<class ObjectType, class FieldType>
	inline FieldInfo FieldInfo::SetSetter(void(ObjectType::* setter)(FieldType))
	{
		this->setter = MethodBind::Create(reinterpret_cast<void(Object::*)(FieldType)>(setter));
		return *this;
	}

	/*template<class ObjectType, class FieldType>
	inline FieldInfo::FieldInfo(const std::string& name, FieldType ObjectType::* field, const BindingType& type, char* hintData) : name(name), bind(FieldBind::Create(reinterpret_cast<FieldType Object::*>(field))), type(type), objectType(0)
	{
		
	}

	template<class ObjectType, class FieldType>
	inline FieldInfo::FieldInfo(const std::string& name, FieldType ObjectType::* field, const BindingType& type, const std::size_t& objectType) : name(name), bind(FieldBind::Create(reinterpret_cast<FieldType Object::*>(field))), type(type), objectType(objectType), hintData(nullptr)
	{
	}

	template<class ObjectType, class FieldType>
	inline FieldInfo::FieldInfo(const std::string & name, FieldType ObjectType::* field, void(ObjectType::* setter)(FieldType), const BindingType & type, const std::size_t & objectType)
	{
	}*/

	/*template<class ObjectType, class FieldType>
	inline FieldInfo::FieldInfo(const std::string& name, FieldType ObjectType::* field, void(ObjectType::* setter)(FieldType), const BindingType& type, const std::size_t& objectType) :
		name(name),
		bind(FieldBind::Create(reinterpret_cast<FieldType Object::*>(field))),
		setter(),
		type(type),
		objectType(objectType),
		hintData(nullptr)
	{
	}*/
}