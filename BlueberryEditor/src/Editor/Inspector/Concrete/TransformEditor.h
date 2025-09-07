#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class TransformEditor : public ObjectEditor
	{
	public:
		virtual ~TransformEditor() = default;

		virtual void OnEnable() override;
		virtual void OnDrawInspector() override;
		virtual void OnDrawSceneSelected() override;

	private:
		SerializedProperty m_LocalPositionProperty;
		SerializedProperty m_LocalRotationProperty;
		SerializedProperty m_LocalScaleProperty;
		SerializedProperty m_LocalRotationEulerHintProperty;
	};
}