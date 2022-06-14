#pragma once

class Scene;
class Entity;

class SceneInspector
{
public:
	SceneInspector() = default;
	SceneInspector(const Ref<Scene>& scene);

	void DrawUI();
	void DrawEntity(const Ref<Entity>& entity);

private:
	Ref<Scene> m_Scene;
};