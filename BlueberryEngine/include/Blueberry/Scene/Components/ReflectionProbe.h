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

	class BB_API ReflectionProbe : public Component
	{
		OBJECT_DECLARATION(ReflectionProbe)

	public:
		ReflectionProbe() = default;
		virtual ~ReflectionProbe() = default;

		ReflectionProbeType GetType();
		void SetType(ReflectionProbeType type);

		uint32_t GetAtlasIndex() const;
		void SetAtlasIndex(uint32_t atlasIndex);

		float GetRadius() const;
		const Vector3& GetSize() const;

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