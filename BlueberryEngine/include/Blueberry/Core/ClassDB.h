#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Core\MethodBind.h"
#include "Blueberry\Tools\StringHelper.h"
#include "Blueberry\Serialization\Enums.h"

namespace Blueberry
{
	enum class BB_API BindingType
	{
		None = 0,

		Bool,
		Int,
		Uint,
		Long,
		Ulong,
		Float,
		String,
		ByteData,

		IntList,
		FloatList,

		StringList,

		Enum,

		Vector2,
		Vector2Int,
		Vector3,
		Vector3Int,
		Vector4,
		Vector4Int,
		Quaternion,
		Color,
		AABB,
		Matrix,

		Vector2List,
		Vector3List,
		Vector4List,
		QuaternionList,
		MatrixList,

		// POD class of fixed size
		Raw,

		// Derived from Object
		ObjectPtr,
		ObjectPtrList,

		// Not derived from Object
		Data,
		DataList,

		Variant
	};

	inline bool IsList(const BindingType& type)
	{
		return (type == BindingType::ObjectPtrList || type == BindingType::DataList || type == BindingType::StringList || type == BindingType::IntList || type == BindingType::FloatList);
	}

	enum class BB_API VisibilityType
	{
		Visible,
		Hidden,
		NonExposed
	};

	struct BB_API FieldOptions
	{
		FieldOptions& SetEnumHint(char* hintData);
		FieldOptions& SetObjectType(TypeId* type);
		FieldOptions& SetSize(uint32_t size);
		FieldOptions& SetVisibility(VisibilityType visibility);
		FieldOptions& SetSerializationFlags(SerializationFlags serializationFlags);
		FieldOptions& SetUpdateCallback(MethodBind* updateCallback);

		TypeId* objectType;
		uint32_t size;
		void* hintData;
		VisibilityType visibility;
		SerializationFlags serializationFlags = SerializationFlags::EditorAndRuntime;
		MethodBind* updateCallback;
	};

	struct BB_API FieldInfo
	{
		String name;
		uint32_t offset;
		BindingType type;
		FieldOptions options;
		bool isList;

		template<class Type>
		Type* Get(void* target) const;

		template<class Type>
		void Set(void* target, Type value) const;
	};

	struct BB_API ClassInfo
	{
		String name;
		TypeId id;
		TypeId parentId;
		Object*(*createInstance)() = nullptr;
		bool isObject;
		bool isDll;
		bool preferBinary;
		bool executeAlways;
		size_t offset;
		List<FieldInfo> fields;
		Dictionary<String, FieldInfo> fieldsMap;
		List<TypeId*> iterators;

		const FieldInfo* GetField(const String& name) const
		{
			for (auto& field : fields)
			{
				if (field.name == name)
				{
					return &field;
				}
			}
			return nullptr;
		}

		Object* Create() const
		{
			Object* object = createInstance();
			ObjectDB::AllocateId(object);
			return object;
		}

		Object* Create(const ObjectId& id) const
		{
			Object* object = createInstance();
			object->m_ObjectId = id;
			return object;
		}
	};

	class BB_API ClassDB
	{
	public:
		static const TypeId GetTypeId(const String& name);
		static const TypeId GetTypeId(const size_t& nameHash);
		static const ClassInfo* GetInfo(const String& name);
		static const ClassInfo* GetInfo(const size_t& nameHash);
		static const ClassInfo* GetInfo(const TypeId& id);
		static List<ClassInfo>& GetInfos();
		static bool IsParent(const TypeId& id, const TypeId& parentId);

		template<class ObjectType>
		static void Register();
		template<class ObjectType>
		static void RegisterAbstract();
		template <class ObjectType>
		static void RegisterData();
		template <class ObjectType>
		static void RegisterIterator();
		template<class ObjectType>
		static void Bind();

		static void DefineField(FieldInfo info);
		static void DefineIterator(TypeId* type);
		static void DefinePreferBinary(); // TODO class attributes
		static void DefineExecuteAlways();

		template <class ObjectType, class BaseObjectType>
		static void DefineBaseFields();

	private:
		template<class ObjectType>
		static Object* CreateObject()
		{
			return new ObjectType();
		}
		static TypeId GetOrCreateTypeId(const String& name);

	private:
		static List<ClassInfo> s_Classes;
		static Dictionary<size_t, TypeId> s_NameToTypeId;
		static List<FieldInfo> s_CurrentFieldInfos;
		static ClassInfo s_CurrentClassInfo;
		static uint32_t s_CurrentOffset;
	};

	constexpr auto GetFieldName(std::string_view name)
	{
		return name.substr(name.find_last_of(':') + 1);
	}
	
	#define REGISTER_CLASS( classname ) ClassDB::Register<classname>();
	#define REGISTER_ABSTRACT_CLASS( classname ) ClassDB::RegisterAbstract<classname>();
	#define REGISTER_DATA_CLASS( classname ) ClassDB::RegisterData<classname>();
	#define REGISTER_ITERATOR( classname ) ClassDB::RegisterIterator<classname>();

	#define DEFINE_BASE_FIELDS( className, baseClassName ) ClassDB::DefineBaseFields<className, baseClassName>();
	#define DEFINE_FIELD( className, fieldName, fieldType, fieldOptions ) ClassDB::DefineField({ TO_STRING(fieldName), offsetof(className, className::fieldName), fieldType, fieldOptions, IsList(fieldType) });
	#define DEFINE_ITERATOR( className ) ClassDB::DefineIterator(&className::Type);
	#define DEFINE_PREFER_BINARY() ClassDB::DefinePreferBinary();
	#define DEFINE_EXECUTE_ALWAYS() ClassDB::DefineExecuteAlways();

	template<class ObjectType>
	inline void ClassDB::Register()
	{
		TypeId id = GetOrCreateTypeId(ObjectType::TypeName);
		TypeId parentId = GetOrCreateTypeId(ObjectType::ParentTypeName);
		String name = ObjectType::TypeName;
		size_t offset = reinterpret_cast<char*>(static_cast<Object*>(reinterpret_cast<ObjectType*>(0x10000000))) - reinterpret_cast<char*>(0x10000000);

		ObjectType::Type = id;
		ObjectType::ParentType = parentId;

		ClassInfo info = {};
		info.name = name;
		info.id = id;
		info.parentId = parentId;
		info.createInstance = &ClassDB::CreateObject<ObjectType>;
		info.isObject = true;
#if BUILD_DLL
		info.isDll = true;
#endif
		info.offset = offset;
		s_CurrentClassInfo = info;

		ObjectType::DefineFields();
		ClassDB::Bind<ObjectType>();
	}

	template<class ObjectType>
	inline void ClassDB::RegisterAbstract()
	{
		TypeId id = GetOrCreateTypeId(ObjectType::TypeName);
		TypeId parentId = ObjectType::ParentTypeName.empty() ? 0 : GetOrCreateTypeId(ObjectType::ParentTypeName);
		String name = ObjectType::TypeName;
		size_t offset = reinterpret_cast<char*>(static_cast<Object*>(reinterpret_cast<ObjectType*>(0x10000000))) - reinterpret_cast<char*>(0x10000000);

		ObjectType::Type = id;
		ObjectType::ParentType = parentId;

		ClassInfo info = {};
		info.name = name;
		info.id = id;
		info.parentId = parentId;
		info.createInstance = nullptr;
		info.isObject = true;
#if BUILD_DLL
		info.isDll = true;
#endif
		info.offset = offset;
		s_CurrentClassInfo = info;

		ObjectType::DefineFields();
		ClassDB::Bind<ObjectType>();
	}

	template<class ObjectType>
	inline void ClassDB::RegisterData()
	{
		TypeId id = GetOrCreateTypeId(ObjectType::TypeName);
		TypeId parentId = 0;
		String name = ObjectType::TypeName;
		size_t offset = reinterpret_cast<char*>(static_cast<Data*>(reinterpret_cast<ObjectType*>(0x10000000))) - reinterpret_cast<char*>(0x10000000);

		ObjectType::Type = id;

		ClassInfo info = {};
		info.name = name;
		info.id = id;
		info.parentId = parentId;
		info.createInstance = nullptr;
		info.isObject = false;
#if BUILD_DLL
		info.isDll = true;
#endif
		info.offset = offset;
		s_CurrentClassInfo = info;

		ObjectType::DefineFields();
		ClassDB::Bind<ObjectType>();
	}

	template<class ObjectType>
	inline void ClassDB::RegisterIterator()
	{
		TypeId id = GetOrCreateTypeId(ObjectType::TypeName);
		TypeId parentId = 0;
		String name = ObjectType::TypeName;

		ClassInfo info = {};
		info.name = name;
		info.id = id;
		info.parentId = parentId;
		info.createInstance = nullptr;
		info.isObject = false;
#if BUILD_DLL
		info.isDll = true;
#endif
		info.offset = 0;

		s_Classes.resize(id + 1);
		s_Classes[id] = std::move(info);
		s_NameToTypeId.insert_or_assign(TO_HASH(name), id);
	}

	template<class ObjectType>
	inline void ClassDB::Bind()
	{
		TypeId id = ObjectType::Type;
		ClassInfo info = s_CurrentClassInfo;
		for (FieldInfo fieldInfo : s_CurrentFieldInfos)
		{
			info.fields.push_back(fieldInfo);
			info.fieldsMap.insert_or_assign(fieldInfo.name, fieldInfo);
		}
		s_Classes.resize(id + 1);
		s_Classes[id] = std::move(info);
		s_NameToTypeId.insert_or_assign(TO_HASH(ObjectType::TypeName), id);
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

	template<class Type>
	inline Type* FieldInfo::Get(void* target) const
	{
		return reinterpret_cast<Type*>(static_cast<char*>(target) + offset);
	}

	template<class Type>
	inline void FieldInfo::Set(void* target, Type value) const
	{
		*reinterpret_cast<Type*>(static_cast<char*>(target) + offset) = value;
	}
}