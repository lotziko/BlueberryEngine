#include "AssemblySerializer.h"

#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	void AssemblySerializer::Serialize()
	{
		List<Object*> objects;
		for (Object* object : ObjectDB::GetObjects(Object::Type))
		{
			const ClassInfo* info = ClassDB::GetInfo(object->GetType());
			if (info->isDll)
			{
				SerializationTree tree = {};
				tree.typeId = object->GetType();
				tree.typeName = ClassDB::GetInfo(tree.typeId)->name;
				tree.fileId = GetFileId(object->GetObjectId());
				tree.objectId = object->GetObjectId();
				tree.isText = false;
				SerializeNode(tree.GetRoot(), Context::Create(object, object->GetType()), SerializationFlags::EditorAndRuntime);
				// TODO handle components OnDisable()/OnDestroy() for now just ignore them because DEFINE_EXECUTE_ALWAYS() is rare
				objects.push_back(object);
				m_Trees.push_back(tree);
			}
		}
		for (Object* object : objects)
		{
			delete object;
		}
	}

	void AssemblySerializer::Deserialize()
	{
		for (auto& tree : m_Trees)
		{
			const ClassInfo* info = ClassDB::GetInfo(tree.typeName);
			Object* object = info->Create(tree.objectId);
			ObjectDB::IdToObjectItem(tree.objectId)->object = object;
			DeserializeNode(tree.GetConstRoot(), Context::Create(object, object->GetType()));
			// TODO handle components OnCreate()/OnEnable()
		}
	}
}
