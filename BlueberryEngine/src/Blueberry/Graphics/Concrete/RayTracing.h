#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Scene;
	class Camera;
	class RayTracingShader;
	class GfxTexture;
	class GfxBuffer;
	class GfxTopLevelAccelerationStructure;

	class RayTracing
	{
	public:
		static void Initialize();
		static void Shutdown();
		static void Draw(Scene* scene, Camera* camera, GfxTexture* output, Rectangle viewport, Vector2Int size);

	private:
		static RayTracingShader* s_Shader;
		static GfxBuffer* s_ConstantBuffer;
		static GfxTopLevelAccelerationStructure* s_AccelerationStructure;
	};
}