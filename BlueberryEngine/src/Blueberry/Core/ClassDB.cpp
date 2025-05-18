#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	Dictionary<size_t, ClassDB::ClassInfo> ClassDB::s_Classes = {};
	List<FieldInfo> ClassDB::s_CurrentFieldInfos = {};
	uint32_t ClassDB::s_CurrentOffset = 0;

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

	void ClassDB::DefineField(FieldInfo info)
	{
		info.offset += s_CurrentOffset;
		s_CurrentFieldInfos.emplace_back(std::move(info));
	}

	FieldOptions& FieldOptions::SetEnumHint(char* hintData)
	{
		auto names = new List<String>();
		StringHelper::Split(hintData, ',', *names);
		this->hintData = names;
		return *this;
	}

	FieldOptions& FieldOptions::SetObjectType(const size_t& type)
	{
		this->objectType = type;
		return *this;
	}

	FieldOptions& FieldOptions::SetSize(const uint32_t& size)
	{
		this->size = size;
		return *this;
	}

	FieldOptions& FieldOptions::SetHidden()
	{
		isHidden = true;
		return *this;
	}
}