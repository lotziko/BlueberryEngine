#include "RegisterAudioTypes.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Audio\AudioClip.h"

namespace Blueberry
{
	void RegisterAudioTypes()
	{
		REGISTER_CLASS(AudioClip);
	}
}