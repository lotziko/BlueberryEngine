#pragma once

#include <string>

namespace Blueberry
{
	class Object;

	class ClassDB
	{
	public:
		struct ClassInfo
		{
			std::string name;
			std::size_t parentId;
			std::function<Ref<Object>()> createInstance;
		};

		template <class T>
		static Ref<Object> CreateInstance()
		{
			return CreateRef<T>();
		}

		static std::map<std::size_t, ClassInfo>& GetInfos();
		static bool IsParent(const std::size_t& id, const std::size_t& parentId);

		static void Register(const std::size_t& id, const std::size_t& parentId, const std::string& name, const std::function<Ref<Object>()>&& createFunction);

	private:
		static std::map<std::size_t, ClassInfo> s_Classes;
	};

	#define REGISTER_CLASS( classname ) ClassDB::Register(classname::Type, classname::ParentType, #classname, &ClassDB::CreateInstance<classname>);
	#define REGISTER_ABSTRACT_CLASS( classname ) ClassDB::Register(classname::Type, classname::ParentType, #classname, nullptr);
}