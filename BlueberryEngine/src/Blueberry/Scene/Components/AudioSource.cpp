#include "Blueberry\Scene\Components\AudioSource.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Audio\AudioClip.h"
#include "Blueberry\Audio\Audio.h"

#include <miniaudio\miniaudio.h>

namespace Blueberry
{
	OBJECT_DEFINITION(AudioSource, Component)
	{
		DEFINE_BASE_FIELDS(AudioSource, Component)
		DEFINE_FIELD(AudioSource, m_AudioClip, BindingType::ObjectPtr, FieldOptions().SetObjectType(&AudioClip::Type))
		DEFINE_FIELD(AudioSource, m_Volume, BindingType::Float, FieldOptions())
		DEFINE_FIELD(AudioSource, m_IsLoop, BindingType::Bool, FieldOptions())
	}

	void AudioSource::OnDestroy()
	{
		if (m_Sound != nullptr)
		{
			ma_audio_buffer_uninit(static_cast<ma_audio_buffer*>(m_Buffer));
			ma_sound_uninit(static_cast<ma_sound*>(m_Sound));
			BB_FREE(m_Buffer);
			BB_FREE(m_Sound);
		}
	}

	void AudioSource::OnEnable()
	{
		if (m_Sound == nullptr && m_AudioClip.IsValid())
		{
			if (m_AudioClip->m_Pcm.size() == 0)
			{
				return;
			}

			ma_audio_buffer_config bufferConfig = ma_audio_buffer_config_init(ma_format_f32, m_AudioClip->m_Channels, m_AudioClip->m_Pcm.size() / m_AudioClip->m_Channels, m_AudioClip->m_Pcm.data(), NULL);
			ma_audio_buffer* buffer = static_cast<ma_audio_buffer*>(BB_MALLOC(sizeof(ma_audio_buffer)));
			ma_result result = ma_audio_buffer_init(&bufferConfig, buffer);

			if (result != MA_SUCCESS)
			{
				BB_ERROR("Failed to create audio buffer.");
				BB_FREE(buffer);
				m_Buffer = nullptr;
				return;
			}
			m_Buffer = buffer;

			ma_sound* sound = static_cast<ma_sound*>(BB_MALLOC(sizeof(ma_sound)));
			result = ma_sound_init_from_data_source(Audio::s_Engine, buffer, 0, NULL, sound);

			if (result != MA_SUCCESS)
			{
				BB_ERROR("Failed to initialize sound.");
				BB_FREE(buffer);
				BB_FREE(sound);
				m_Buffer = nullptr;
				m_Sound = nullptr;
				return;
			}
			m_Sound = sound;
		}

		if (m_Sound != nullptr)
		{
			ma_sound* sound = static_cast<ma_sound*>(m_Sound);
			ma_sound_set_volume(sound, std::clamp(m_Volume, 0.0f, 1.0f));
			ma_sound_set_looping(sound, m_IsLoop);
			ma_sound_start(sound);
		}
	}

	void AudioSource::OnDisable()
	{
		if (m_Sound != nullptr)
		{
			ma_sound* sound = static_cast<ma_sound*>(m_Sound);
			ma_sound_stop(sound);
		}
	}
}