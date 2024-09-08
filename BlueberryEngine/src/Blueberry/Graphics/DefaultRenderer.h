#pragma once

namespace Blueberry
{
	class Scene;
	class Camera;
	class RenderTexture;
	class Material;

	class DefaultRenderer
	{
	public:
		static void Initialize();
		static void Shutdown();

		static void Draw(Scene* scene, Camera* camera, Rectangle viewport, Color background, RenderTexture* output);

		static RenderTexture* GetColorMSAA();
		static RenderTexture* GetDepthStencilMSAA();

		static RenderTexture* GetColor();
		static RenderTexture* GetDepthStencil();

	private:
		static inline RenderTexture* s_ColorMSAARenderTarget = nullptr;
		static inline RenderTexture* s_DepthStencilMSAARenderTarget = nullptr;

		static inline RenderTexture* s_ColorRenderTarget = nullptr;
		static inline RenderTexture* s_DepthStencilRenderTarget = nullptr;

		static inline Material* s_ResolveMSAAMaterial = nullptr;
	};
}