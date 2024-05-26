#include "bbpch.h"
#include "ClassDB.h"

namespace Blueberry
{
	std::unordered_map<std::size_t, ClassDB::ClassInfo> ClassDB::s_Classes = std::unordered_map<std::size_t, ClassDB::ClassInfo>();

	const ClassDB::ClassInfo& ClassDB::GetInfo(const std::size_t& id)
	{
		return s_Classes.find(id)->second;
	}

	std::unordered_map<std::size_t, ClassDB::ClassInfo>& ClassDB::GetInfos()
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

	FieldInfo FieldInfo::SetHintData(char* hintData)
	{
		if (type == BindingType::Enum)
		{
			auto names = new std::vector<std::string>();
			StringHelper::Split(hintData, ',', *names);
			this->hintData = names;
		}
		else
		{
			this->hintData = hintData;
		}
		return *this;
	}

	FieldInfo FieldInfo::SetObjectType(const std::size_t& objectType)
	{
		this->objectType = objectType;
		return *this;
	}
}