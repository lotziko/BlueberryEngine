#include "Blueberry\Audio\AudioClip.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Audio\Audio.h"

#include <miniaudio\miniaudio.h>

namespace Blueberry
{
	OBJECT_DEFINITION(AudioClip, Object)
	{
		DEFINE_BASE_FIELDS(AudioClip, Object)
	}

	void AudioClip::Initialize(const ByteData& data)
	{
		ma_decoder decoder;
		ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 0, 0);
		ma_result result = ma_decoder_init_memory(data.data(), data.size(), &config, &decoder);

		if (result != MA_SUCCESS)
		{
			BB_ERROR("Failed to create decoder.");
			return;
		}

		ma_uint64 frameCount;
		ma_decoder_get_length_in_pcm_frames(&decoder, &frameCount);
		
		m_Channels = decoder.outputChannels;
		m_Pcm.resize(frameCount * m_Channels);
		ma_decoder_read_pcm_frames(&decoder, m_Pcm.data(), frameCount, NULL);
	}
}