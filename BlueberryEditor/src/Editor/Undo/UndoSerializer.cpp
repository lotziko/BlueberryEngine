#include "UndoSerializer.h"

namespace Blueberry
{
	void UndoSerializer::Serialize(const List<Object*>& objects)
	{
		for (Object* object : objects)
		{
			SerializationTree tree = {};
			tree.typeId = object->GetType();
			tree.typeName = ClassDB::GetInfo(tree.typeId)->name;
			tree.fileId = GetFileId(object->GetObjectId());
			tree.objectId = object->GetObjectId();
			tree.isText = false;
			SerializeNode(tree.GetRoot(), Context::Create(object, object->GetType()), SerializationFlags::EditorAndRuntime);
			m_Trees.push_back(tree);
		}
	}

	void UndoSerializer::Deserialize()
	{
		for (auto& tree : m_Trees)
		{
			const ClassInfo* info = ClassDB::GetInfo(tree.typeName);
			Object* object = info->Create(tree.objectId);
			ObjectDB::IdToObjectItem(tree.objectId)->object = object;
		}
		for (auto& tree : m_Trees)
		{
			Object* object = ObjectDB::GetObject(tree.objectId);
			DeserializeNode(tree.GetConstRoot(), Context::Create(object, object->GetType()));
		}
	}

	bool UndoSerializer::IsEmpty() const
	{
		return m_Trees.empty();
	}
}