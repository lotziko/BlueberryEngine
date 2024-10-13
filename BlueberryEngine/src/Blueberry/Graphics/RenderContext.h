#pragma once

#include "GfxDrawingOperation.h"

namespace Blueberry
{
	class Scene;
	class Camera;
	class Light;
	class Material;
	class MeshRenderer;

	struct CullingResults
	{
		struct DrawingOperation
		{
			Matrix matrix;
			Mesh* mesh;
			UINT submeshIndex;
			Material* material;
		};

		Camera* camera;
		std::vector<Light*> lights;
		std::vector<MeshRenderer*> meshRenderers;
		std::vector<DrawingOperation> drawingOperations;
	};

	struct DrawingSettings
	{
		uint8_t passIndex;
	};

	class RenderContext
	{
	public:
		void Cull(Scene* scene, Camera* camera, CullingResults& results);
		void Bind(CullingResults& results);
		void Draw(CullingResults& results, DrawingSettings& drawingSettings);

	private:
		static inline GfxVertexBuffer* s_IndexBuffer = nullptr;
	};
}