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
		int lightmapSize;
		Vector3Int probeVolumeSize;
		bool denoise;
	};

	struct CalculationResult
	{
		bool hasLightmap;
		List<uint8_t> lightmapOutput;
		Vector2Int lightmapOutputSize;
		List<Vector4> chartOffsetScale;
		Dictionary<ObjectId, uint32_t> chartInstanceOffset;

		bool hasProbes;
		ByteData probeOutput;
		uint32_t probeOutputSize;
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
		static void BakeLightmap(Scene* scene, const CalculationParams& params);
		static void BakeProbes(Scene* scene, const CalculationParams& params, const List<Vector3>& positions);
		static void Shutdown();

		static float GetProgress();
		static const LightmappingState& GetLightmappingState();
		static CalculationResult& GetCalculationResult();

	private:
		static LightmappingState s_LightmappingState;
	};
}