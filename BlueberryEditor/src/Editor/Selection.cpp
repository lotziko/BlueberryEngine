#include "bbpch.h"
#include "Selection.h"

namespace Blueberry
{
	std::set<ObjectId> Selection::s_ActiveObjects = std::set<ObjectId>();

	Object* Selection::GetActiveObject()
	{
		if (s_ActiveObjects.size() > 0)
		{
			ObjectId first = *s_ActiveObjects.begin();
			return ObjectDB::GetObject(first);
		}
		return nullptr;
	}

	bool Selection::IsActiveObject(Object* object)
	{
		return s_ActiveObjects.count(object->GetObjectId());
	}

	void Selection::AddActiveObject(Object* object)
	{
		s_ActiveObjects.insert(object->GetObjectId());
	}

	void Selection::SetActiveObject(Object* object)
	{
		s_ActiveObjects.clear();
		if (object != nullptr)
		{
			s_ActiveObjects.insert(object->GetObjectId());
		}
	}
}