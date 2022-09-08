#include "bbpch.h"
#include "Selection.h"

namespace Blueberry
{
	Object* Selection::s_ActiveObject = nullptr;

	Object* Selection::GetActiveObject()
	{
		return s_ActiveObject;
	}

	void Selection::SetActiveObject(Object* object)
	{
		s_ActiveObject = object;
	}
}