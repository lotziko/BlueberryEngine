#pragma once

#include "Blueberry\Core\Base.h"

struct ma_engine;

namespace Blueberry
{
	class AudioClip;

	class BB_API Audio
	{
	public:
		static void Initialize();
		static void Shutdown();
		static void Update();
		static void Play(AudioClip* clip, float volume = 1.0f, float pitch = 1.0f);

	private:
		static ma_engine* s_Engine;

		friend class AudioClip;
		friend class AudioSource;
	};
}