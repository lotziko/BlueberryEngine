#pragma once

#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Events\Event.h"

namespace Blueberry
{
	using SelectionChangeEvent = Event<>;

	class Selection
	{
	public:
		static Object* GetActiveObject();
		static bool IsActiveObject(Object* object);
		static void AddActiveObject(Object* object);
		static void SetActiveObject(Object* object);

		static SelectionChangeEvent& GetSelectionChanged();

	private:
		static std::unordered_set<ObjectId> s_ActiveObjects;
		static SelectionChangeEvent s_SelectionChanged;
	};
}