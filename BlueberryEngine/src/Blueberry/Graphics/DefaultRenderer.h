#pragma once

namespace Blueberry
{
	class Scene;
	class Camera;
	class RenderTexture;
	class Material;
	class HBAORenderer;
	class ShadowAtlas;

	class TextureCube;

	class DefaultRenderer
	{
	public:
		static void Initialize();
		static void Shutdown();
		
		static void Draw(Scene* scene, Camera* camera, Rectangle viewport, Color background, RenderTexture* colorOutput = nullptr, RenderTexture* depthOutput = nullptr, const bool& simplified = false);
		
	private:
		static inline Material* s_ResolveMSAAMaterial = nullptr;
		static inline ShadowAtlas* s_ShadowAtlas = nullptr;
	};
}