#pragma once

class Scene;
class Camera;

class SceneRenderer
{
public:
	static void Draw(const Ref<Scene>& scene);
	static void Draw(const Ref<Scene>& scene, Camera* camera);
	static void Draw(const Ref<Scene>& scene, const Matrix& viewMatrix, const Matrix& projectionMatrix);
};