#pragma once

#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Serialization\Serializer.h"

#include "..\UndoSerializer.h"
#include "..\UndoRecord.h"

namespace Blueberry
{
	class Entity;
	class Component;

	class UndoAddEntityRecord : public UndoRecord
	{
	public:
		UndoAddEntityRecord(Entity* entity);

		virtual void OnUndo() final;
		virtual void OnRedo() final;

	private:
		ObjectPtr<Entity> m_Entity;
		UndoSerializer m_Serializer = {};
	};
}