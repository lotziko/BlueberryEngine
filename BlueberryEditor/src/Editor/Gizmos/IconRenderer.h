#pragma once

namespace Blueberry
{
	class Scene;
	class BaseCamera;
	class Texture2D;
	class Material;

	class IconRenderer
	{
	public:
		static bool Initialize();
		static void Shutdown();

		static void Draw(Scene* scene, BaseCamera* camera);

	private:
		static inline Material* s_IconMaterial = nullptr;
	};
}