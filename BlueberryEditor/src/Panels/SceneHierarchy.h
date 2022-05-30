#pragma once

class GraphicsDevice;
class Scene;
class Entity;

class SceneHierarchy
{
public:
	SceneHierarchy() = default;
	SceneHierarchy(const Ref<Scene>& scene);

	void DrawUI();
	void DrawEntity(const Ref<Entity>& entity);

private:
	Ref<Scene> m_Scene;
};