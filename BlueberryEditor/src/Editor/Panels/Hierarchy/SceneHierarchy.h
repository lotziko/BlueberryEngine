#pragma once

#include "Editor\Panels\EditorWindow.h"

namespace Blueberry
{
	class Entity;

	class SceneHierarchy : public EditorWindow
	{
		OBJECT_DECLARATION(SceneHierarchy)

	public:
		SceneHierarchy() = default;
		virtual ~SceneHierarchy() = default;

		static void Open();

		virtual void OnDrawUI() final;

	private:
		void DrawEntity(Entity* entity);
		void DrawCreateEntity();
		void DrawDestroyEntity(Entity*& entity);
		void DrawUnpackPrefabEntity(Entity* entity);

		Entity* m_ActiveEntity;
	};
}