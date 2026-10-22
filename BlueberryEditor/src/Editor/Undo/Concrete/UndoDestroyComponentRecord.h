#pragma once

#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Serialization\Serializer.h"

#include "..\UndoSerializer.h"
#include "..\UndoRecord.h"

namespace Blueberry
{
	class Entity;
	class Component;

	class UndoDestroyComponentRecord : public UndoRecord
	{
	public:
		UndoDestroyComponentRecord(Component* component);

		virtual void OnUndo() final;
		virtual void OnRedo() final;

	private:
		ObjectPtr<Entity> m_Entity;
		ObjectPtr<Component> m_Component;
		size_t m_Index = 0;
		UndoSerializer m_Serializer = {};
	};
}