#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class Entity;
	class Component;

	class EntityEditor : public ObjectEditor
	{
	public:
		virtual ~EntityEditor() = default;

		virtual void OnEnable() override;
		virtual void OnDisable() override;
		virtual void OnDrawInspector() override;

	private:
		void OnEntityDestroy();

	private:
		SerializedProperty m_IsActiveProperty;
		SerializedProperty m_ComponentsProperty;
		List<std::pair<Object*, ObjectEditor*>> m_ComponentsEditors;
		List<std::pair<Entity*, Component*>> m_AddedComponents;
		List<Component*> m_RemovedComponents;
	};
}