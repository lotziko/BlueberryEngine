#include "Blueberry\Animations\AnimationClip.h"

#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	DATA_DEFINITION(AnimationBoneData)
	{
		DEFINE_FIELD(AnimationBoneData, m_Name, BindingType::String, {})
		DEFINE_FIELD(AnimationBoneData, m_Positions, BindingType::Vector3List, {})
		DEFINE_FIELD(AnimationBoneData, m_Rotations, BindingType::QuaternionList, {})
		DEFINE_FIELD(AnimationBoneData, m_Scales, BindingType::Vector3List, {})
	}

	OBJECT_DEFINITION(AnimationClip, Object)
	{
		DEFINE_BASE_FIELDS(AnimationClip, Object)
		DEFINE_FIELD(AnimationClip, m_AnimationBones, BindingType::DataList, FieldOptions().SetObjectType(AnimationBoneData::Type).SetVisibility(VisibilityType::NonExposed))
		DEFINE_FIELD(AnimationClip, m_FrameRate, BindingType::Float, {})
		DEFINE_FIELD(AnimationClip, m_Length, BindingType::Float, {})
	}

	void AnimationBoneData::SetName(const String& name)
	{
		m_Name = name;
	}

	void AnimationBoneData::SetPositions(const Vector3* positions, const size_t& count)
	{
		m_Positions.resize(count);
		memcpy(m_Positions.data(), positions, sizeof(Vector3) * count);
	}

	void AnimationBoneData::SetRotations(const Quaternion* rotations, const size_t& count)
	{
		m_Rotations.resize(count);
		memcpy(m_Rotations.data(), rotations, sizeof(Quaternion) * count);
	}

	void AnimationBoneData::SetScales(const Vector3* scales, const size_t& count)
	{
		m_Scales.resize(count);
		memcpy(m_Scales.data(), scales, sizeof(Vector3) * count);
	}

	void AnimationClip::ClearAnimationBones()
	{
		m_AnimationBones.clear();
	}

	void AnimationClip::AddAnimationBone(const AnimationBoneData& data)
	{
		m_AnimationBones.push_back(std::move(data));
	}

	uint32_t AnimationClip::GetBoneIndex(const String& name)
	{
		for (size_t i = 0; i < m_AnimationBones.size(); ++i)
		{
			if (m_AnimationBones[i].m_Name == name)
			{
				return static_cast<uint32_t>(i);
			}
		}
		return UINT32_MAX;
	}

	size_t AnimationClip::GetBoneCount()
	{
		return m_AnimationBones.size();
	}

	TRS AnimationClip::GetTRS(const float& time, const size_t& index)
	{
		if (index < m_AnimationBones.size())
		{
			auto& bone = m_AnimationBones[index];
			float frame = std::fmodf(time * m_FrameRate, bone.m_Positions.size());
			size_t frameIndex = static_cast<size_t>(std::floorf(frame));
			size_t nextIndex = std::min(bone.m_Positions.size() - 1, frameIndex + 1);
			float t = frame - frameIndex;

			Vector3 position = Vector3::Lerp(bone.m_Positions[frameIndex], bone.m_Positions[nextIndex], t);
			Quaternion rotation = Quaternion::Lerp(bone.m_Rotations[frameIndex], bone.m_Rotations[nextIndex], t);
			Vector3 scale = Vector3::Lerp(bone.m_Scales[frameIndex], bone.m_Scales[nextIndex], t);

			return { position, rotation, scale };
		}
		return {};
	}

	const float& AnimationClip::GetFrameRate()
	{
		return m_FrameRate;
	}

	void AnimationClip::SetFrameRate(const float& frameRate)
	{
		m_FrameRate = frameRate;
	}

	const float& AnimationClip::GetLength()
	{
		return m_Length;
	}

	void AnimationClip::SetLength(const float& length)
	{
		m_Length = length;
	}
}