#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class CharacterControllerEditor : public ObjectEditor
	{
	public:
		virtual ~CharacterControllerEditor() = default;

		virtual void OnDrawSceneSelected() override;
	};
}