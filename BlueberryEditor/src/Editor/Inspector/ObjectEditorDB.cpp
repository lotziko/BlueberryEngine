#include "ObjectEditorDB.h"
#include "ObjectEditor.h"

#include "Blueberry\Graphics\Shader.h"

namespace Blueberry
{
	Dictionary<TypeId, ObjectEditorInfo> ObjectEditorDB::s_ObjectEditors = {};

	Dictionary<TypeId, ObjectEditorInfo>& Blueberry::ObjectEditorDB::GetInfos()
	{
		return s_ObjectEditors;
	}

	const ObjectEditorInfo* ObjectEditorDB::GetInfo(const TypeId& id)
	{
		TypeId inheritsId = id;
		while (true)
		{
			auto objectEditorIt = s_ObjectEditors.find(inheritsId);
			if (objectEditorIt != s_ObjectEditors.end())
			{
				return &objectEditorIt->second;
			}
			inheritsId = ClassDB::GetInfo(inheritsId)->parentId;
			if (inheritsId == 0)
			{
				break;
			}
		}
		return {};
	}
}