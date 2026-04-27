#include "Blueberry\Audio\Audio.h"

#include "Blueberry\Audio\AudioClip.h"

#include <miniaudio\miniaudio.h>

namespace Blueberry
{
	ma_engine* Audio::s_Engine = nullptr;
	List<std::pair<ma_audio_buffer*, ma_sound*>> s_ActiveClips;

	void Audio::Initialize()
	{
		if (s_Engine == nullptr)
		{
			s_Engine = static_cast<ma_engine*>(BB_MALLOC(sizeof(ma_engine)));
			ma_result result = ma_engine_init(NULL, s_Engine);
			if (result != MA_SUCCESS)
			{
				BB_ERROR("Failed to initialize audio engine.");
				return;
			}
		}
	}

	void Audio::Shutdown()
	{
		if (s_Engine != nullptr)
		{
			ma_engine_uninit(s_Engine);
			BB_FREE(s_Engine);
			s_Engine = nullptr;
		}
	}

	void Audio::Update()
	{
		for (auto it = s_ActiveClips.begin(); it != s_ActiveClips.end();)
		{
			if (it->second->atEnd)
			{
				ma_audio_buffer_uninit(it->first);
				ma_sound_uninit(it->second);
				BB_FREE(it->first);
				BB_FREE(it->second);
				it = s_ActiveClips.erase(it);
			}
			else
			{
				++it;
			}
		}
	}

	void Audio::Play(AudioClip* clip, float volume, float pitch)
	{
		if (clip->m_Pcm.size() == 0)
		{
			return;
		}

		ma_audio_buffer_config bufferConfig = ma_audio_buffer_config_init(ma_format_f32, clip->m_Channels, clip->m_Pcm.size() / clip->m_Channels, clip->m_Pcm.data(), NULL);
		ma_audio_buffer* buffer = static_cast<ma_audio_buffer*>(BB_MALLOC(sizeof(ma_audio_buffer)));
		ma_result result = ma_audio_buffer_init(&bufferConfig, buffer);

		if (result != MA_SUCCESS)
		{
			BB_ERROR("Failed to create audio buffer.");
			BB_FREE(buffer);
			return;
		}

		ma_sound* sound = static_cast<ma_sound*>(BB_MALLOC(sizeof(ma_sound)));
		result = ma_sound_init_from_data_source(Audio::s_Engine, buffer, 0, NULL, sound);

		if (result != MA_SUCCESS)
		{
			BB_ERROR("Failed to initialize sound.");
			BB_FREE(buffer);
			BB_FREE(sound);
			return;
		}

		ma_sound_set_volume(sound, volume);
		ma_sound_set_pitch(sound, pitch);
		ma_sound_start(sound);
		s_ActiveClips.push_back(std::make_pair(buffer, sound));
	}
}