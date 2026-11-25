#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class ProbeVolume;

	class ProbeVolumeEditor : public ObjectEditor
	{
	public:
		ProbeVolumeEditor();
		virtual ~ProbeVolumeEditor() = default;

		virtual void OnEnable() final;
		virtual void OnDisable() final;
		virtual void OnDrawInspector() final;

		virtual Texture* GetIcon(Object* object) final;
		virtual void OnDrawSceneSelected() final;

	private:
		SerializedProperty m_SizeProperty;
		SerializedProperty m_SpacingProperty;
	};
}