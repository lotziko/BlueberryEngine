#pragma once

class Scene;
class Camera;

class SceneArea
{
public:
	SceneArea() = default;
	SceneArea(const Ref<Scene>& scene);

	void Draw();
	void SetCamera(Camera* camera) { m_Camera = camera; }
	void SetViewport(const Rect& viewport) { m_Viewport = viewport; }

private:
	Ref<Scene> m_Scene;
	Camera* m_Camera;
	Rect m_Viewport;
};