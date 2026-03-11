#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	List<ClassInfo> ClassDB::s_Classes = {};
	Dictionary<size_t, TypeId> ClassDB::s_NameToTypeId = {};
	List<FieldInfo> ClassDB::s_CurrentFieldInfos = {};
	ClassInfo ClassDB::s_CurrentClassInfo = {};
	uint32_t ClassDB::s_CurrentOffset = 0;

	const TypeId ClassDB::GetTypeId(const String& name)
	{
		auto it = s_NameToTypeId.find(TO_HASH(name));
		if (it != s_NameToTypeId.end())
		{
			return it->second;
		}
		return 0;
	}

	const TypeId ClassDB::GetTypeId(const size_t& nameHash)
	{
		auto it = s_NameToTypeId.find(nameHash);
		if (it != s_NameToTypeId.end())
		{
			return it->second;
		}
		return 0;
	}

	const ClassInfo* ClassDB::GetInfo(const String& name)
	{
		auto it = s_NameToTypeId.find(TO_HASH(name));
		if (it != s_NameToTypeId.end())
		{
			return &s_Classes[it->second];
		}
		return nullptr;
	}

	const ClassInfo* ClassDB::GetInfo(const size_t& nameHash)
	{
		auto it = s_NameToTypeId.find(nameHash);
		if (it != s_NameToTypeId.end())
		{
			return &s_Classes[it->second];
		}
		return nullptr;
	}

	const ClassInfo* ClassDB::GetInfo(const TypeId& id)
	{
		if (id > 0 && id < s_Classes.size())
		{
			return &s_Classes[id];
		}
		return nullptr;
	}

	List<ClassInfo>& ClassDB::GetInfos()
	{
		return s_Classes;
	}

	bool ClassDB::IsParent(const TypeId& id, const TypeId& parentId)
	{
		TypeId inheritsId = id;
		while (true)
		{
			if (inheritsId == parentId)
			{
				return true;
			}
			ClassInfo& info = s_Classes[inheritsId];
			if (info.parentId == 0)
			{
				return false;
			}
			inheritsId = info.parentId;
		}
		return false;
	}

	void ClassDB::DefineField(FieldInfo info)
	{
		info.offset += s_CurrentOffset;
		s_CurrentFieldInfos.push_back(std::move(info));
	}

	void ClassDB::DefineIterator(const TypeId& type)
	{
		s_CurrentClassInfo.iterators.push_back(type);
	}

	void ClassDB::DefinePreferBinary()
	{
		s_CurrentClassInfo.preferBinary = true;
	}

	void ClassDB::DefineExecuteAlways()
	{
		s_CurrentClassInfo.executeAlways = true;
	}

	TypeId ClassDB::GetOrCreateTypeId(const String& name)
	{
		auto it = s_NameToTypeId.find(TO_HASH(name));
		if (it != s_NameToTypeId.end())
		{
			return it->second;
		}
		return GenerateTypeId();
	}

	FieldOptions& FieldOptions::SetEnumHint(char* hintData)
	{
		auto names = new List<String>();
		StringHelper::Split(hintData, ',', *names);
		this->hintData = names;
		return *this;
	}

	FieldOptions& FieldOptions::SetObjectType(TypeId* type)
	{
		this->objectType = type;
		return *this;
	}

	FieldOptions& FieldOptions::SetSize(const uint32_t& size)
	{
		this->size = size;
		return *this;
	}

	FieldOptions& FieldOptions::SetVisibility(const VisibilityType& visibility)
	{
		this->visibility = visibility;
		return *this;
	}

	FieldOptions& FieldOptions::SetUpdateCallback(MethodBind* updateCallback)
	{
		this->updateCallback = updateCallback;
		return *this;
	}
}