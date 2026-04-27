#pragma once

#include "Component.h"

namespace Blueberry
{
	class AudioClip;

	class BB_API AudioSource : public Component
	{
		OBJECT_DECLARATION(AudioSource)

	public:
		AudioSource() = default;
		virtual ~AudioSource() = default;

		virtual void OnDestroy() final;
		virtual void OnEnable() final;
		virtual void OnDisable() final;

	private:
		ObjectPtr<AudioClip> m_AudioClip;
		float m_Volume = 1.0f;
		bool m_IsLoop = true;

		void* m_Buffer;
		void* m_Sound;
	};
}