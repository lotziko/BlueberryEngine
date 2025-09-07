#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\MethodBind.h"
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

		IntList,
		FloatList,

		StringList,

		Enum,

		Vector2,
		Vector3,
		Vector4,
		Quaternion,
		Color,
		AABB,

		// POD class of fixed size stored in objectType
		Raw,

		// Derived from Object
		ObjectPtr,
		ObjectPtrList,

		// Not derived from Object
		Data,
		DataList
	};

	inline bool IsList(const BindingType& type)
	{
		return (type == BindingType::ObjectPtrList || type == BindingType::DataList || type == BindingType::StringList || type == BindingType::IntList || type == BindingType::FloatList);
	}

	enum class VisibilityType
	{
		Visible,
		Hidden,
		NonExposed
	};

	struct FieldOptions
	{
		FieldOptions& SetEnumHint(char* hintData);
		FieldOptions& SetObjectType(const size_t& type);
		FieldOptions& SetSize(const uint32_t& size);
		FieldOptions& SetVisibility(const VisibilityType& visibility);
		FieldOptions& SetUpdateCallback(MethodBind* updateCallback);

		size_t objectType;
		uint32_t size;
		void* hintData;
		VisibilityType visibility;
		MethodBind* updateCallback;
	};

	struct FieldInfo
	{
		String name;
		uint32_t offset;
		BindingType type;
		FieldOptions options;
		bool isList;
	};

	struct ClassInfo
	{
		String name;
		size_t parentId;
		Object*(*createInstance)() = nullptr;
		bool isObject;
		size_t offset;
		List<FieldInfo> fields;
		Dictionary<String, FieldInfo> fieldsMap;
	};

	class BB_API ClassDB
	{
	public:
		static const ClassInfo& GetInfo(const size_t&);
		static Dictionary<size_t, ClassInfo>& GetInfos();
		static bool IsParent(const size_t& id, const size_t& parentId);

		template<class ObjectType>
		static void Register();
		template<class ObjectType>
		static void RegisterAbstract();
		template <class ObjectType>
		static void RegisterData();
		template<class ObjectType>
		static void Bind();

		static void DefineField(FieldInfo info);

		template <class ObjectType, class BaseObjectType>
		static void DefineBaseFields();

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
		static Dictionary<size_t, ClassInfo> s_Classes;
		static List<FieldInfo> s_CurrentFieldInfos;
		static uint32_t s_CurrentOffset;
	};

	constexpr auto GetFieldName(std::string_view name)
	{
		return name.substr(name.find_last_of(':') + 1);
	}
	
	#define REGISTER_CLASS( classname ) ClassDB::Register<classname>();
	#define REGISTER_ABSTRACT_CLASS( classname ) ClassDB::RegisterAbstract<classname>();
	#define REGISTER_DATA_CLASS( classname ) ClassDB::RegisterData<classname>();

	#define DEFINE_BASE_FIELDS( className, baseClassName ) ClassDB::DefineBaseFields<className, baseClassName>();
	#define DEFINE_FIELD( className, fieldName, fieldType, fieldOptions ) ClassDB::DefineField({ TO_STRING(fieldName), offsetof(className, className::fieldName), fieldType, fieldOptions, IsList(fieldType) });

	template<class ObjectType>
	inline void ClassDB::Register()
	{
		size_t id = ObjectType::Type;
		size_t parentId = ObjectType::ParentType;
		String name = ObjectType::TypeName;
		Object*(*createFunction)() = &ClassDB::CreateObject<ObjectType>;
		size_t offset = reinterpret_cast<char*>(static_cast<Object*>(reinterpret_cast<ObjectType*>(0x10000000))) - reinterpret_cast<char*>(0x10000000);

		ClassInfo info = {};
		info.name = name;
		info.parentId = parentId;
		info.createInstance = createFunction;
		info.isObject = true;
		info.offset = offset;

		s_Classes.insert_or_assign(id, info);

		ObjectType::DefineFields();
		ClassDB::Bind<ObjectType>();
	}

	template<class ObjectType>
	inline void ClassDB::RegisterAbstract()
	{
		size_t id = ObjectType::Type;
		size_t parentId = ObjectType::ParentType;
		String name = ObjectType::TypeName;
		size_t offset = reinterpret_cast<char*>(static_cast<Object*>(reinterpret_cast<ObjectType*>(0x10000000))) - reinterpret_cast<char*>(0x10000000);

		ClassInfo info = {};
		info.name = name;
		info.parentId = parentId;
		info.createInstance = nullptr;
		info.isObject = true;
		info.offset = offset;

		s_Classes.insert_or_assign(id, info);

		ObjectType::DefineFields();
		ClassDB::Bind<ObjectType>();
	}

	template<class ObjectType>
	inline void ClassDB::RegisterData()
	{
		size_t id = ObjectType::Type;
		size_t parentId = 0;
		String name = ObjectType::TypeName;
		size_t offset = reinterpret_cast<char*>(static_cast<Data*>(reinterpret_cast<ObjectType*>(0x10000000))) - reinterpret_cast<char*>(0x10000000);

		ClassInfo info = {};
		info.name = name;
		info.parentId = parentId;
		info.createInstance = nullptr;
		info.isObject = false;
		info.offset = offset;

		s_Classes.insert_or_assign(id, info);

		ObjectType::DefineFields();
		ClassDB::Bind<ObjectType>();
	}

	template<class ObjectType>
	inline void ClassDB::Bind()
	{
		auto classInfoIt = s_Classes.find(ObjectType::Type);
		if (classInfoIt != s_Classes.end())
		{
			for (FieldInfo info : s_CurrentFieldInfos)
			{
				classInfoIt->second.fields.emplace_back(info);
				classInfoIt->second.fieldsMap.insert_or_assign(info.name, info);
			}
		}
		s_CurrentFieldInfos.clear();
	}

	template<class ObjectType, class BaseObjectType>
	inline void ClassDB::DefineBaseFields()
	{
		uint32_t offset = static_cast<uint32_t>(reinterpret_cast<char*>(static_cast<BaseObjectType*>(reinterpret_cast<ObjectType*>(0x10000000))) - reinterpret_cast<char*>(0x10000000));
		s_CurrentOffset += offset;
		BaseObjectType::DefineFields();
		s_CurrentOffset -= offset;
	}
}