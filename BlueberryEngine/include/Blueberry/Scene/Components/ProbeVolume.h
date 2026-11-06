#pragma once

#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	struct BB_API ProbeVolumeBrickData : public Data
	{
		DATA_DECLARATION(ProbeVolumeBrickData)

	public:
		void SetColorOffset(uint32_t index, uint32_t offset);

	private:
		uint32_t m_ColorOffsets[64];
		uint32_t m_Children[27];
	};

	class BB_API ProbeVolumeInfo : public Object
	{
		OBJECT_DECLARATION(ProbeVolumeInfo)

	public:
		ProbeVolumeInfo() = default;
		virtual ~ProbeVolumeInfo() = default;

		const Vector3Int& GetSize() const;
		void SetSize(const Vector3Int& size);

		const List<ProbeVolumeBrickData>& GetBricks() const;
		void SetBricks(const List<ProbeVolumeBrickData>& bricks);

		const ByteData& GetColors() const;
		void SetColors(const ByteData& colors);

	private:
		Vector3Int m_Size;
		List<ProbeVolumeBrickData> m_Bricks;
		ByteData m_Colors;
	};

	class BB_API ProbeVolume : public Component
	{
		OBJECT_DECLARATION(ProbeVolume)

	public:
		ProbeVolume() = default;
		virtual ~ProbeVolume() = default;

		const Vector3& GetSize() const;
		void SetSize(const Vector3& size);

		Vector3Int GetDensity() const;

		float GetSpacing() const;
		void SetSpacing(float spacing);

		uint32_t GetIndex() const;
		void SetIndex(uint32_t index);

	private:
		Vector3 m_Size = Vector3(1, 1, 1);
		float m_Spacing = 1.0f;

		uint32_t m_Index = UINT32_MAX;
		uint32_t m_ProbeAtlasOffset = UINT32_MAX;

		friend class BakedLighting;
		friend class PerCameraLightDataConstantBuffer;
	};
}