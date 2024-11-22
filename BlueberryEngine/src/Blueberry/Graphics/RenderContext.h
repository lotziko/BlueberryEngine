#pragma once

#include "GfxDrawingOperation.h"

namespace Blueberry
{
	class Scene;
	class Renderer;
	class Camera;
	class Light;
	class Material;
	class MeshRenderer;

	struct CullingResults
	{
		struct RendererInfo
		{
			Renderer* renderer;
			size_t type;
			AABB bounds;
			uint32_t frustumMask;
		};

		struct CullerInfo
		{
			Object* object;
			uint8_t index;
			DirectX::XMVECTOR nearPlane, farPlane, rightPlane, leftPlane, topPlane, bottomPlane;
		};

		Camera* camera;
		std::vector<Light*> lights;
		std::vector<RendererInfo> rendererInfos;
		std::vector<CullerInfo> cullerInfos;
	};

	struct DrawingSettings
	{
		uint8_t passIndex;
	};

	struct ShadowDrawingSettings
	{
		Light* light;
		uint8_t sliceIndex;
	};

	class RenderContext
	{
	public:
		void Cull(Scene* scene, Camera* camera, CullingResults& results);
		void BindCamera(CullingResults& results);
		void DrawShadows(CullingResults& results, ShadowDrawingSettings& shadowDrawingSettings);
		void DrawRenderers(CullingResults& results, DrawingSettings& drawingSettings);

	private:
		static inline GfxVertexBuffer* s_IndexBuffer = nullptr;
	};
}