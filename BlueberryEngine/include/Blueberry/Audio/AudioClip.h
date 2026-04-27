#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class BB_API AudioClip : public Object
	{
		OBJECT_DECLARATION(AudioClip)

	public:
		AudioClip() = default;
		virtual ~AudioClip() = default;

		void Initialize(const ByteData& data);

	private:
		List<float> m_Pcm;
		uint32_t m_Channels = 0;

		friend class Audio;
		friend class AudioSource;
	};
}