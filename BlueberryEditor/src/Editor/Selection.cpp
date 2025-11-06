#include "Selection.h"

namespace Blueberry
{
	HashSet<ObjectId> Selection::s_ActiveObjects = {};
	SelectionChangeEvent Selection::s_SelectionChanged = {};

	Object* Selection::GetActiveObject()
	{
		if (s_ActiveObjects.size() > 0)
		{
			ObjectId first = *s_ActiveObjects.begin();
			return ObjectDB::GetObject(first);
		}
		return nullptr;
	}

	// TODO calculate on changing selection instead
	List<Object*> Selection::GetActiveObjects()
	{
		List<Object*> result;
		for (ObjectId id : s_ActiveObjects)
		{
			result.emplace_back(ObjectDB::GetObject(id));
		}
		return result;
	}

	bool Selection::IsActiveObject(Object* object)
	{
		return s_ActiveObjects.count(object->GetObjectId());
	}

	void Selection::AddActiveObject(Object* object)
	{
		if (object == nullptr || s_ActiveObjects.count(object->GetObjectId()) > 0)
		{
			return;
		}
		s_ActiveObjects.insert(object->GetObjectId());
		s_SelectionChanged.Invoke();
	}

	void Selection::SetActiveObject(Object* object)
	{
		if (object != nullptr && s_ActiveObjects.size() == 1 && *s_ActiveObjects.begin() == object->GetObjectId())
		{
			return;
		}
		if (object == nullptr && s_ActiveObjects.size() == 0)
		{
			return;
		}
		s_ActiveObjects.clear();
		if (object != nullptr)
		{
			s_ActiveObjects.insert(object->GetObjectId());
		}
		s_SelectionChanged.Invoke();
	}

	SelectionChangeEvent& Selection::GetSelectionChanged()
	{
		return s_SelectionChanged;
	}
}