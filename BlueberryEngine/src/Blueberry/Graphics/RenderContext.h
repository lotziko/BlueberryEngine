#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "GfxDrawingOperation.h"

namespace Blueberry
{
	class Scene;
	class Renderer;
	class Camera;
	class Light;
	class Material;
	class MeshRenderer;

	struct CameraData
	{
		Camera* camera;
		Vector2Int size;
		Vector2Int renderTargetSize;
		Matrix multiviewViewMatrix[2];
		Matrix multiviewProjectionMatrix[2];
		Rectangle multiviewViewport;
		bool isMultiview;
	};

	struct CullingResults
	{
		struct CullerInfo
		{
			Object* object;
			uint8_t index;
			DirectX::XMVECTOR planes[6];
			List<ObjectId> renderers;
			Matrix viewMatrix;
		};

		Camera* camera;
		List<Light*> lights;
		List<CullerInfo> cullerInfos;
	};

	enum class SortingMode
	{
		Default,
		FrontToBack
	};

	struct DrawingSettings
	{
		uint8_t passIndex;
		SortingMode sortingMode;
	};

	struct ShadowDrawingSettings
	{
		Light* light;
		uint8_t sliceIndex;
	};

	class RenderContext
	{
	public:
		void Cull(Scene* scene, CameraData& cameraData, CullingResults& results);
		void BindCamera(CullingResults& results, CameraData& cameraData);
		void DrawSky(Scene* scene);
		void DrawShadows(CullingResults& results, ShadowDrawingSettings& shadowDrawingSettings);
		void DrawRenderers(CullingResults& results, DrawingSettings& drawingSettings);

	private:
		static inline GfxBuffer* s_IndexBuffer = nullptr;
		static inline size_t s_LastCullingFrame = 0;
	};
}