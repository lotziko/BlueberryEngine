#include "bbpch.h"
#include "ClassDB.h"

namespace Blueberry
{
	Dictionary<size_t, ClassDB::ClassInfo> ClassDB::s_Classes = {};

	const ClassDB::ClassInfo& ClassDB::GetInfo(const size_t& id)
	{
		return s_Classes.find(id)->second;
	}

	Dictionary<size_t, ClassDB::ClassInfo>& ClassDB::GetInfos()
	{
		return s_Classes;
	}

	bool ClassDB::IsParent(const size_t& id, const size_t& parentId)
	{
		size_t inheritsId = id;
			
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
			auto names = new List<std::string>();
			StringHelper::Split(hintData, ',', *names);
			this->hintData = names;
		}
		else
		{
			this->hintData = hintData;
		}
		return *this;
	}

	FieldInfo FieldInfo::SetObjectType(const size_t& objectType)
	{
		this->objectType = objectType;
		return *this;
	}
}