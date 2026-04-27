#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class AnimationTransitionEditor : public ObjectEditor
	{
	public:
		virtual ~AnimationTransitionEditor() = default;
		
		virtual void OnEnable() override;
		virtual void OnDrawInspector() override;

	private:
		SerializedProperty m_IsFixedDurationProperty;
		SerializedProperty m_TransitionDurationProperty;
		SerializedProperty m_TransitionOffsetProperty;
		SerializedProperty m_HasExitTimeProperty;
		SerializedProperty m_ExitTimeProperty;
		SerializedProperty m_ConditionsProperty;
	};
}