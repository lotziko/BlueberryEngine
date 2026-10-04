#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class AudioSourceEditor : public ObjectEditor
	{
	public:
		AudioSourceEditor();
		virtual ~AudioSourceEditor() = default;

		virtual Texture* GetIcon(Object* object) final;
	};
}