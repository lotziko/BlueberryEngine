#pragma once

#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Serialization\Serializer.h"

#include "..\UndoSerializer.h"
#include "..\UndoRecord.h"

namespace Blueberry
{
	class Entity;
	class Component;

	class UndoAddComponentRecord : public UndoRecord
	{
	public:
		UndoAddComponentRecord(Entity* entity, Component* component);

		virtual void OnUndo() final;
		virtual void OnRedo() final;

	private:
		ObjectPtr<Entity> m_Entity;
		ObjectPtr<Component> m_Component;
		UndoSerializer m_Serializer = {};
	};
}