#include "Blueberry\Scene\Components\ProbeVolume.h"

#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	DATA_DEFINITION(ProbeVolumeBrickData)
	{
		DEFINE_FIELD(ProbeVolumeBrickData, m_ColorOffsets, BindingType::Raw, FieldOptions().SetSize(sizeof(uint32_t) * 64))
		DEFINE_FIELD(ProbeVolumeBrickData, m_Children, BindingType::Raw, FieldOptions().SetSize(sizeof(uint32_t) * 27))
	}

	OBJECT_DEFINITION(ProbeVolumeInfo, Object)
	{
		DEFINE_FIELD(ProbeVolumeInfo, m_Size, BindingType::Vector3Int, FieldOptions())
		DEFINE_FIELD(ProbeVolumeInfo, m_Bricks, BindingType::DataList, FieldOptions().SetObjectType(&ProbeVolumeBrickData::Type))
		DEFINE_FIELD(ProbeVolumeInfo, m_Colors, BindingType::ByteData, FieldOptions())
		DEFINE_PREFER_BINARY()
	}

	OBJECT_DEFINITION(ProbeVolume, Component)
	{
		DEFINE_BASE_FIELDS(ProbeVolume, Component)
		DEFINE_FIELD(ProbeVolume, m_Size, BindingType::Vector3, FieldOptions())
		DEFINE_FIELD(ProbeVolume, m_Spacing, BindingType::Float, FieldOptions())
		DEFINE_FIELD(ProbeVolume, m_Index, BindingType::Uint, FieldOptions().SetSerializationFlags(SerializationFlags::RuntimeOnly))
		DEFINE_ITERATOR(ProbeVolume)
		DEFINE_EXECUTE_ALWAYS()
		DEFINE_ADD_COMPONENT_MENU("Rendering/Probe Volume")
	}

	void ProbeVolumeBrickData::SetColorOffset(uint32_t index, uint32_t offset)
	{
		m_ColorOffsets[index] = offset;
	}

	const Vector3Int& ProbeVolumeInfo::GetSize() const
	{
		return m_Size;
	}

	void ProbeVolumeInfo::SetSize(const Vector3Int& size)
	{
		m_Size = size;
	}

	const List<ProbeVolumeBrickData>& ProbeVolumeInfo::GetBricks() const
	{
		return m_Bricks;
	}

	void ProbeVolumeInfo::SetBricks(const List<ProbeVolumeBrickData>& bricks)
	{
		m_Bricks = bricks;
	}

	const ByteData& ProbeVolumeInfo::GetColors() const
	{
		return m_Colors;
	}

	void ProbeVolumeInfo::SetColors(const ByteData& colors)
	{
		m_Colors = colors;
	}

	const Vector3& ProbeVolume::GetSize() const
	{
		return m_Size;
	}

	void ProbeVolume::SetSize(const Vector3& size)
	{
		m_Size = size;
	}

	Vector3Int ProbeVolume::GetDensity() const
	{
		if (m_Spacing > 0)
		{
			return Vector3Int(static_cast<int>(std::floorf(m_Size.x / m_Spacing)), static_cast<int>(std::floorf(m_Size.y / m_Spacing)), static_cast<int>(std::floorf(m_Size.z / m_Spacing)));
		}
		return Vector3Int(1, 1, 1);
	}

	float ProbeVolume::GetSpacing() const
	{
		return m_Spacing;
	}

	void ProbeVolume::SetSpacing(float spacing)
	{
		m_Spacing = spacing;
	}

	uint32_t ProbeVolume::GetIndex() const
	{
		return m_Index;
	}

	void ProbeVolume::SetIndex(uint32_t index)
	{
		m_Index = index;
	}
}