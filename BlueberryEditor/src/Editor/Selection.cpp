#include "bbpch.h"
#include "Selection.h"

namespace Blueberry
{
	ObjectPtr<Object> Selection::s_ActiveObject = nullptr;

	Object* Selection::GetActiveObject()
	{
		return s_ActiveObject.Get();
	}

	void Selection::SetActiveObject(Object* object)
	{
		s_ActiveObject = object;
	}
}