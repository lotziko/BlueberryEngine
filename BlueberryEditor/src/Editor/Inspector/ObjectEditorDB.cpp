#include "ObjectEditorDB.h"
#include "ObjectEditor.h"

#include "Blueberry\Graphics\Shader.h"

namespace Blueberry
{
	Dictionary<size_t, ObjectEditorInfo> ObjectEditorDB::s_ObjectEditors = {};

	Dictionary<size_t, ObjectEditorInfo>& Blueberry::ObjectEditorDB::GetInfos()
	{
		return s_ObjectEditors;
	}

	const ObjectEditorInfo& ObjectEditorDB::GetInfo(const size_t& id)
	{
		size_t inheritsId = id;
		while (true)
		{
			auto objectEditorIt = s_ObjectEditors.find(inheritsId);
			if (objectEditorIt != s_ObjectEditors.end())
			{
				return objectEditorIt->second;
			}
			inheritsId = ClassDB::GetInfo(inheritsId).parentId;
			if (inheritsId == 0)
			{
				break;
			}
		}
		return {};
	}
}