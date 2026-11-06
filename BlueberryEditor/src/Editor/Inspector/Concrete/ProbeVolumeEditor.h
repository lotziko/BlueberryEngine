#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class ProbeVolumeEditor : public ObjectEditor
	{
	public:
		ProbeVolumeEditor();
		virtual ~ProbeVolumeEditor() = default;

		virtual void OnEnable() override;
		virtual void OnDrawInspector() override;

		virtual Texture* GetIcon(Object* object) final;
		virtual void OnDrawSceneSelected() override;

	private:
		SerializedProperty m_SizeProperty;
		SerializedProperty m_SpacingProperty;
	};
}