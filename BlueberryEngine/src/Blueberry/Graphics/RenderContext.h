#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Graphics\GfxDrawingOperation.h"

namespace Blueberry
{
	class Scene;
	class Renderer;
	class Camera;
	class SkyRenderer;
	class ProbeVolume;
	class ReflectionProbe;
	class Light;
	class Canvas;
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
		SkyRenderer* skyRenderer;
		ProbeVolume* probeVolume;
		List<ReflectionProbe*> reflectionProbes;
		List<Light*> lights;
		List<Canvas*> canvases;
		List<CullerInfo> cullerInfos;
	};

	enum class SortingMode
	{
		Default,
		FrontToBack
	};

	enum class ObjectsFilter
	{
		All,
		Dynamic,
		Static
	};

	struct DrawingSettings
	{
		uint8_t passIndex;
		SortingMode sortingMode;
		ObjectsFilter objectsFilter;
	};

	struct ShadowDrawingSettings
	{
		Light* light;
		uint32_t sliceIndex;
		ObjectsFilter objectsFilter;
	};

	class RenderContext
	{
	public:
		void Cull(Scene* scene, CameraData& cameraData, CullingResults& results);
		void BindCamera(CullingResults& results, CameraData& cameraData);
		void DrawSky(CullingResults& results);
		void DrawShadows(CullingResults& results, ShadowDrawingSettings& shadowDrawingSettings);
		void DrawRenderers(CullingResults& results, DrawingSettings& drawingSettings);
		void DrawCanvases(CullingResults& results);

	private:
		static GfxBuffer* s_IndexBuffer;
		static size_t s_LastCullingFrame;
	};
}