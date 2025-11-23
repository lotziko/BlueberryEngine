#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class ReflectionProbeEditor : public ObjectEditor
	{
	public:
		ReflectionProbeEditor();
		virtual ~ReflectionProbeEditor() = default;

		virtual void OnEnable() override;
		virtual void OnDrawInspector() override;

		virtual Texture* GetIcon(Object* object) final;
		virtual void OnDrawSceneSelected() override;

	private:
		SerializedProperty m_SizeProperty;
	};
}