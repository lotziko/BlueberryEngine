#pragma once

#include "Blueberry\Core\ObjectPtr.h"

#include "..\UndoSerializer.h"
#include "..\UndoRecord.h"

namespace Blueberry
{
	class Entity;
	class Transform;
	class PrefabInstance;
	class Scene;

	class UndoDestroyEntityRecord : public UndoRecord
	{
	public:
		UndoDestroyEntityRecord(Entity* entity);
		~UndoDestroyEntityRecord();

		virtual void OnUndo() final;
		virtual void OnRedo() final;

	private:
		ObjectPtr<Entity> m_Entity;
		ObjectPtr<Transform> m_Parent;
		List<ObjectPtr<PrefabInstance>> m_PrefabInstances;
		size_t m_SiblingIndex;
		UndoSerializer m_Serializer = {};
	};
}