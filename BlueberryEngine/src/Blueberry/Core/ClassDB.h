#pragma once

#include <string>
#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class ClassDB
	{
	public:
		struct ClassInfo
		{
			std::string name;
			std::size_t parentId;
			std::function<Ref<Object>()> createInstance;
		};

		static std::map<std::size_t, ClassInfo>& GetInfos();
		static bool IsParent(const std::size_t& id, const std::size_t& parentId);

		static void Register(const std::size_t& id, const std::size_t& parentId, const std::string& name, const std::function<Ref<Object>()>&& createFunction);

	private:
		static std::map<std::size_t, ClassInfo> s_Classes;
	};

	#define REGISTER_CLASS( classname ) ClassDB::Register(classname::Type, classname::ParentType, #classname, &ObjectDB::CreateObject<classname>);
	#define REGISTER_ABSTRACT_CLASS( classname ) ClassDB::Register(classname::Type, classname::ParentType, #classname, nullptr);
}