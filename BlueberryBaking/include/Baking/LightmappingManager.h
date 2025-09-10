#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Events\Event.h"

namespace Blueberry
{
	class Scene;
	class Camera;

	struct CalculationParams
	{
		int tileSize;
		float texelPerUnit;
		int samplePerTexel;
		int maxSize;
		bool denoise;
	};

	struct CalculationResult
	{
		List<uint8_t> output;
		Vector2Int outputSize;
		List<Vector4> chartOffsetScale;
		Dictionary<ObjectId, uint32_t> chartInstanceOffset;
	};

	enum class LightmappingState
	{
		None,
		Calculating,
		Waiting
	};

	class LightmappingManager
	{
	public:
		static void Clear();
		static void Calculate(Scene* scene, Camera* camera, const Vector2Int& viewport, uint8_t* output);
		static void Calculate(Scene* scene, const CalculationParams& params);
		static void Shutdown();

		static float GetProgress();
		static const LightmappingState& GetLightmappingState();
		static CalculationResult& GetCalculationResult();

	private:
		static LightmappingState s_LightmappingState;
	};
}