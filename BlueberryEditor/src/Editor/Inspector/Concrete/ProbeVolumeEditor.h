#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class ProbeVolumeEditor : public ObjectEditor
	{
	public:
		ProbeVolumeEditor() = default;
		virtual ~ProbeVolumeEditor() = default;

		virtual void OnDrawSceneSelected() override;
	};
}