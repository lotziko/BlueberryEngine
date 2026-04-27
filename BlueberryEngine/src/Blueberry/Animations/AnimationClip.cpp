#include "Blueberry\Animations\AnimationClip.h"

#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	DATA_DEFINITION(AnimationBoneData)
	{
		DEFINE_FIELD(AnimationBoneData, m_Name, BindingType::String, FieldOptions())
		DEFINE_FIELD(AnimationBoneData, m_Positions, BindingType::Vector3List, FieldOptions())
		DEFINE_FIELD(AnimationBoneData, m_Rotations, BindingType::QuaternionList, FieldOptions())
		DEFINE_FIELD(AnimationBoneData, m_Scales, BindingType::Vector3List, FieldOptions())
	}

	OBJECT_DEFINITION(AnimationClip, Object)
	{
		DEFINE_BASE_FIELDS(AnimationClip, Object)
		DEFINE_FIELD(AnimationClip, m_AnimationBones, BindingType::DataList, FieldOptions().SetObjectType(&AnimationBoneData::Type).SetVisibility(VisibilityType::NonExposed))
		DEFINE_FIELD(AnimationClip, m_FrameRate, BindingType::Float, FieldOptions())
		DEFINE_FIELD(AnimationClip, m_Length, BindingType::Float, FieldOptions())
		DEFINE_FIELD(AnimationClip, m_IsLoop, BindingType::Bool, FieldOptions())
	}

	void AnimationBoneData::SetName(const String& name)
	{
		m_Name = name;
	}

	void AnimationBoneData::SetPositions(const Vector3* positions, size_t count)
	{
		m_Positions.resize(count);
		memcpy(m_Positions.data(), positions, sizeof(Vector3) * count);
	}

	void AnimationBoneData::SetRotations(const Quaternion* rotations, size_t count)
	{
		m_Rotations.resize(count);
		memcpy(m_Rotations.data(), rotations, sizeof(Quaternion) * count);
	}

	void AnimationBoneData::SetScales(const Vector3* scales, size_t count)
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
		m_AnimationBones.push_back(data);
	}

	uint32_t AnimationClip::GetBoneIndex(const String& name) const
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

	size_t AnimationClip::GetBoneCount() const
	{
		return m_AnimationBones.size();
	}

	TRS AnimationClip::GetTRS(float time, size_t index) const
	{
		if (index < m_AnimationBones.size())
		{
			auto& bone = m_AnimationBones[index];
			float positionCount = static_cast<float>(bone.m_Positions.size());
			float frame = m_IsLoop ? std::fmodf(time * m_FrameRate, positionCount) : std::clamp(time * m_FrameRate, 0.0f, positionCount - 1);
			size_t frameIndex = static_cast<size_t>(std::floorf(frame));
			size_t nextIndex = std::min(bone.m_Positions.size() - 1, frameIndex + 1);
			float t = frame - frameIndex;

			Vector3 position = Vector3::Lerp(bone.m_Positions[frameIndex], bone.m_Positions[nextIndex], t);
			Quaternion rotation = Quaternion::Slerp(bone.m_Rotations[frameIndex], bone.m_Rotations[nextIndex], t);
			Vector3 scale = Vector3::Lerp(bone.m_Scales[frameIndex], bone.m_Scales[nextIndex], t);

			return { position, rotation, scale };
		}
		return {};
	}

	float AnimationClip::GetFrameRate() const
	{
		return m_FrameRate;
	}

	void AnimationClip::SetFrameRate(float frameRate)
	{
		m_FrameRate = frameRate;
	}

	float AnimationClip::GetLength() const
	{
		return m_Length;
	}

	void AnimationClip::SetLength(float length)
	{
		m_Length = length;
	}

	bool AnimationClip::IsLoop() const
	{
		return m_IsLoop;
	}

	void AnimationClip::SetLoop(bool loop)
	{
		m_IsLoop = loop;
	}
}