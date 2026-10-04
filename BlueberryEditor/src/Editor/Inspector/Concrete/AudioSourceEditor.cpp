#include "AudioSourceEditor.h"

#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\Texture.h"

namespace Blueberry
{
	static Texture* s_Icon = nullptr;

	AudioSourceEditor::AudioSourceEditor()
	{
		if (s_Icon == nullptr)
		{
			s_Icon = static_cast<Texture*>(AssetLoader::Load("assets/icons/AudioSourceIcon.png"));
		}
	}

	Texture* AudioSourceEditor::GetIcon(Object* object)
	{
		return s_Icon;
	}
}