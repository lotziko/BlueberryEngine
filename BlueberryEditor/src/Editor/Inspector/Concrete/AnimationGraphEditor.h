#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class AnimationGraphEditor : public ObjectEditor
	{
	public:
		virtual ~AnimationGraphEditor() = default;

		virtual void OnDrawInspector() override;
	};
}