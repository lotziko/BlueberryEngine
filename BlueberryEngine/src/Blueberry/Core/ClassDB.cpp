#include "bbpch.h"
#include "ClassDB.h"

namespace Blueberry
{
	std::map<std::size_t, ClassDB::ClassInfo> ClassDB::s_Classes = std::map<std::size_t, ClassDB::ClassInfo>();

	std::map<std::size_t, ClassDB::ClassInfo>& ClassDB::GetInfos()
	{
		return s_Classes;
	}

	bool ClassDB::IsParent(const std::size_t& id, const std::size_t& parentId)
	{
		std::size_t inheritsId = id;
			
		while (s_Classes.count(inheritsId) > 0)
		{
			if (inheritsId == parentId)
			{
				return true;
			}

			inheritsId = s_Classes.find(inheritsId)->second.parentId;
		}

		return false;
	}

	void ClassDB::Register(const std::size_t& id, const std::size_t& parentId, const std::string& name, const std::function<Ref<Object>()>&& createFunction)
	{
		if (s_Classes.count(id) == 0)
		{
			s_Classes.insert({ id, { name, parentId, createFunction } });
		}
	}
}