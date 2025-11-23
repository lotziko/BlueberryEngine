#pragma once

#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	class TextureCube;

	class ReflectionProbe : public Component
	{
		OBJECT_DECLARATION(ReflectionProbe)

	public:
		ReflectionProbe() = default;
		virtual ~ReflectionProbe() = default;

		virtual void OnEnable() final;
		virtual void OnDisable() final;

		TextureCube* GetReflectionTexture();
		void SetReflectionTexture(TextureCube* texture);

		void SetAtlasIndex(const uint32_t& atlasIndex);

		const Vector3& GetSize();

	private:
		ObjectPtr<TextureCube> m_ReflectionTexture;
		Vector3 m_Size = Vector3(1, 1, 1);

		uint32_t m_AtlasIndex = UINT_MAX;

		friend class ReflectionAtlas;
		friend class PerCameraLightDataConstantBuffer;
	};
}