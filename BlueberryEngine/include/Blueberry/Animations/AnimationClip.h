#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class BB_API AnimationBoneData : public Data
	{
		DATA_DECLARATION(AnimationBoneData)

	public:
		AnimationBoneData() = default;

		void SetName(const String& name);
		void SetPositions(const Vector3* positions, size_t count);
		void SetRotations(const Quaternion* rotations, size_t count);
		void SetScales(const Vector3* scales, size_t count);

	private:
		String m_Name;
		List<Vector3> m_Positions;
		List<Quaternion> m_Rotations;
		List<Vector3> m_Scales;

		friend class AnimationClip;
	};

	class BB_API AnimationClip : public Object
	{
		OBJECT_DECLARATION(AnimationClip)

	public:
		AnimationClip() = default;
		virtual ~AnimationClip() = default;

		void ClearAnimationBones();
		void AddAnimationBone(const AnimationBoneData& data);
		uint32_t GetBoneIndex(const String& name) const;

		size_t GetBoneCount() const;
		TRS GetTRS(float time, size_t index) const;

		float GetFrameRate() const;
		void SetFrameRate(float frameRate);
		
		float GetLength() const;
		void SetLength(float length);

		bool IsLoop() const;
		void SetLoop(bool loop);

	private:
		List<AnimationBoneData> m_AnimationBones;
		float m_FrameRate = 0.0f;
		float m_Length = 0.0f;
		bool m_IsLoop = false;
	};
}