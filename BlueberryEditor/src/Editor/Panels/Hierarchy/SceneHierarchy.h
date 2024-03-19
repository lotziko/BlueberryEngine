#pragma once

namespace Blueberry
{
	class Entity;

	class SceneHierarchy
	{
	public:
		SceneHierarchy() = default;
		virtual ~SceneHierarchy() = default;

		void DrawUI();

	private:
		void DrawEntity(Entity* entity);
		void DrawCreateEntity();
		void DrawDestroyEntity(Entity*& entity);
		void DrawRenameEntity(Entity* entity);
	};
}