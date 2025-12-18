#pragma once

#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	enum class BB_API ReflectionProbeType
	{
		Sphere,
		Box
	};

	class TextureCube;

	class ReflectionProbe : public Component
	{
		OBJECT_DECLARATION(ReflectionProbe)

	public:
		ReflectionProbe() = default;
		virtual ~ReflectionProbe() = default;

		const ReflectionProbeType& GetType();
		void SetType(const ReflectionProbeType& type);

		const uint32_t& GetAtlasIndex();
		void SetAtlasIndex(const uint32_t& atlasIndex);

		const float& GetRadius();
		const Vector3& GetSize();

	private:
		ReflectionProbeType m_Type = ReflectionProbeType::Sphere;
		float m_Radius = 1.0f;
		Vector3 m_Size = Vector3(1, 1, 1);
		float m_Fade = 0.25f;

		uint32_t m_AtlasIndex = UINT_MAX;

		friend class ReflectionAtlas;
		friend class PerCameraLightDataConstantBuffer;
	};
}