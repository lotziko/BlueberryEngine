#pragma once

namespace Blueberry
{
	class Scene;
	class Entity;

	class SceneHierarchy
	{
	public:
		SceneHierarchy() = default;
		SceneHierarchy(const Ref<Scene>& scene);

		void DrawUI();

	private:
		void DrawEntity(Entity* entity);
	private:
		Ref<Scene> m_Scene;
	};
}