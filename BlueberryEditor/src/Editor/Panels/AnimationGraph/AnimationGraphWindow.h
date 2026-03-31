#pragma once

#include "Editor\Panels\EditorWindow.h"

namespace Blueberry
{
	class AnimationGraph;
	class AnimationStateMachine;

	class AnimationGraphWindow : public EditorWindow
	{
		OBJECT_DECLARATION(AnimationGraphWindow)

	public:
		AnimationGraphWindow() = default;
		virtual ~AnimationGraphWindow() = default;

		static void Open();
		static void Open(AnimationGraph* graph);

		virtual void OnDrawUI() final;

		AnimationGraph* GetGraph();

	private:
		void DrawLeftPanel();
		void DrawRightPanel();

		size_t FindStateIndex(size_t pinId);
		size_t FindStateIndex(ObjectId id);
		size_t FindTransitionIndex(ObjectId id);
		void SetGraph(AnimationGraph* graph);

	private:
		enum class StateType
		{
			Default,
			Entry,
			Any
		};

		struct StateData
		{
			ObjectId id;
			StateType type;
			size_t pin;
		};

		struct TransitionData
		{
			ObjectId id;
			size_t fromPin;
			size_t toPin;
		};

		struct PinData
		{
			bool isInput;
		};

		ObjectPtr<AnimationGraph> m_AnimationGraph;
		ObjectPtr<AnimationStateMachine> m_StateMachine;
		List<StateData> m_States;
		List<TransitionData> m_Transitions;
		bool m_IsInitialized = false;
		size_t m_MaxPinId = 1;
	};
}