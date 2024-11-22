#pragma once

namespace Blueberry
{
	class Transform;
	class Light;

	struct LightData
	{
		Transform* transform;
		Light* light;
	};

	class GfxConstantBuffer;

	class PerCameraLightDataConstantBuffer
	{
	public:
		static void BindData(const LightData& mainLight, const std::vector<LightData>& lights);

	private:
		static inline GfxConstantBuffer* s_ConstantBuffer = nullptr;
	};
}