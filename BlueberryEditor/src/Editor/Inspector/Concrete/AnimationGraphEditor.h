#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class GfxTexture;

	class AnimationGraphEditor : public ObjectEditor
	{
	public:
		virtual ~AnimationGraphEditor() = default;

		virtual void OnDrawInspector() override;
	};
}