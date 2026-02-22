#include "AssemblySerializer.h"

#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	void AssemblySerializer::Serialize()
	{
		List<Object*> objects;
		ObjectDB::GetObjects(Object::Type, objects);
		for (Object* object : objects)
		{
			const ClassInfo* info = ClassDB::GetInfo(object->GetType());
			if (info->isDll)
			{
				SerializationTree tree = {};
				tree.type = object->GetType();
				tree.fileId = GetFileId(object->GetObjectId());
				tree.objectId = object->GetObjectId();
				tree.isText = false;
				SerializeNode(tree.GetRoot(), Context::Create(object, object->GetType()));
				// TODO handle components OnDisable()/OnDestroy() for now just ignore them because DEFINE_EXECUTE_ALWAYS() is rare
				delete object;
				m_Trees.push_back(tree);
			}
		}
	}

	void AssemblySerializer::Deserialize()
	{
		for (auto& tree : m_Trees)
		{
			const ClassInfo* info = ClassDB::GetInfo(tree.type);
			Object* object = info->Create(tree.objectId);
			ObjectDB::IdToObjectItem(tree.objectId)->object = object;
			DeserializeNode(tree.GetConstRoot(), Context::Create(object, object->GetType()));
			// TODO handle components OnCreate()/OnEnable()
		}
	}
}
