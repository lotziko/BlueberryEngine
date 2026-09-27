#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Graphics\GfxTexturePool.h"

namespace Blueberry
{
	class Scene;
	class Camera;
	class RayTracingShader;
	class ComputeShader;
	class GfxTexture;
	class GfxBuffer;
	class GfxTopLevelAccelerationStructure;
	class PerCameraData;

	struct PerCameraReflectionsData
	{
		std::unique_ptr<GfxTexture, ReturnTextureToPool> previousViewZ;
		std::unique_ptr<GfxTexture, ReturnTextureToPool> previousNormalRoughness;
		std::unique_ptr<GfxTexture, ReturnTextureToPool> previousInternalData;
		std::unique_ptr<GfxTexture, ReturnTextureToPool> specularHistory;
		std::unique_ptr<GfxTexture, ReturnTextureToPool> specularFastHistory;
		std::unique_ptr<GfxTexture, ReturnTextureToPool> historyStabilizedPing;
		std::unique_ptr<GfxTexture, ReturnTextureToPool> historyStabilizedPong;
		std::unique_ptr<GfxTexture, ReturnTextureToPool> specularHitdistForTrackingPing;
		std::unique_ptr<GfxTexture, ReturnTextureToPool> specularHitdistForTrackingPong;
		std::unique_ptr<GfxTexture, ReturnTextureToPool> output;

		Matrix previousViewToWorld = Matrix::Identity;
		Matrix previousProjection = Matrix::Identity;
		Vector4 previousFrustum = Vector4(0, 0, 0, 0);
		Vector2 previousRectSize = Vector2(0, 0);
		Vector2Int previousResourceSize = Vector2Int(0, 0);
		Vector2 previousJitter = Vector2(0, 0);
		float previousSplitScreen = 0.0f;
		bool previousIsOrthographic = false;
		bool historyValid = false;
	};

	class Reflections
	{
	public:
		static void Initialize();
		static void Shutdown();
		static void Draw(Scene* scene, Camera* camera, Rectangle viewport, Vector2Int size, PerCameraData& perCameraData);
		static GfxTexture* GetReflectionTexture(const PerCameraData& perCameraData);

	private:
		static RayTracingShader* s_ReflectionsRayTracingShader;
		static ComputeShader* s_ReblurComputeShaders[7];
		static GfxBuffer* s_ReflectionsCameraBuffer;
		static GfxBuffer* s_ReblurBuffer;
		static GfxTopLevelAccelerationStructure* s_AccelerationStructure;
	};
}