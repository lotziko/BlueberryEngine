#include "AnimationGraphWindow.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Animations\AnimationGraph.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Menu\EditorMenuManager.h"
#include "Editor\Selection.h"
#include "Editor\Misc\ImGuiHelper.h"

#include <imgui\imgui.h>
#include <imguinode\imgui_node_editor.h>
#include <imguinode\imgui_extra_math.h>

namespace ed = ax::NodeEditor;

namespace Blueberry
{
	OBJECT_DEFINITION(AnimationGraphWindow, EditorWindow)
	{
		DEFINE_BASE_FIELDS(AnimationGraphWindow, EditorWindow)
		DEFINE_FIELD(AnimationGraphWindow, m_AnimationGraph, BindingType::ObjectPtr, FieldOptions().SetObjectType(&AnimationGraph::Type))
		EditorMenuManager::AddItem("Window/AnimationGraph", &AnimationGraphWindow::Open);
	}

	void AnimationGraphWindow::Open()
	{
		EditorWindow* window = GetWindow(AnimationGraphWindow::Type);
		window->SetTitle("Animation Graph");
		window->Show();
	}

	void AnimationGraphWindow::Open(AnimationGraph* graph)
	{
		AnimationGraphWindow* window = static_cast<AnimationGraphWindow*>(GetWindow(AnimationGraphWindow::Type));
		window->SetTitle("Animation Graph");
		window->SetGraph(graph);
		window->Show();
	}

	static ed::EditorContext* s_Context = nullptr;
	static int s_AnyNodeId = -2;
	static int s_EntryNodeId = -1;
	static int s_EntryTransitionId = -1;
	static float s_AlignSize = 16.0f;
	static float s_GridSize = 16.0f;

	static void LinkDrawCallback(ImDrawList* drawList, const ImVec2& from, const ImVec2& to, ImU32 color, float thickness)
	{
		float halfThickness = thickness * 0.5f;
		ImVec2 direction = ImNormalized(to - from);
		ImVec2 tangent = ImVec2(-direction.y, direction.x);
		ImVec2 center = (from + to) * 0.5f;

		drawList->PathLineTo(from);
		drawList->PathLineTo(to);
		drawList->PathStroke(color, 0, thickness);
		drawList->PathLineTo(center + direction * (5 + thickness));
		drawList->PathLineTo(center - direction * 5 + tangent * (5 + halfThickness));
		drawList->PathLineTo(center - direction * 5 - tangent * (5 + halfThickness));
		drawList->PathFillConvex(color);
	}

	static void LinkUpdateEndpointsCallback(ImVec2& from, ImVec2& to, bool isPreview)
	{
		ImVec2 direction = ImNormalized(to - from);
		ImVec2 tangent = ImVec2(-direction.y, direction.x);
		ImVec2 offset = tangent * 7.0f;
		from += offset;
		if (!isPreview)
		{
			to += offset;
		}
	}

	static float ImDistancePointSegment(ImVec2 p, ImVec2 a, ImVec2 b)
	{
		ImVec2 ab = ImVec2(b.x - a.x, b.y - a.y);
		ImVec2 ap = ImVec2(p.x - a.x, p.y - a.y);
		float abLenSq = ab.x * ab.x + ab.y * ab.y;
		float t = (ap.x * ab.x + ap.y * ab.y) / abLenSq;
		t = ImClamp(t, 0.0f, 1.0f);
		ImVec2 closest = ImVec2(a.x + ab.x * t,	a.y + ab.y * t);
		float dx = p.x - closest.x;
		float dy = p.y - closest.y;
		return sqrtf(dx * dx + dy * dy);
	}

	static bool LinkTestHitCallback(const ImVec2& point, const ImVec2& from, const ImVec2& to, float thickness)
	{
		float distance = ImDistancePointSegment(point, from, to);
		return distance <= thickness;
	}

	void AnimationGraphWindow::OnDrawUI()
	{
		if (m_AnimationGraph == nullptr)
		{
			return;
		}

		if (s_Context == nullptr)
		{
			ed::Config config;
			config.SettingsFile = nullptr;
			config.LinkDrawCallback = LinkDrawCallback;
			config.LinkUpdateEndpointsCallback = LinkUpdateEndpointsCallback;
			config.LinkTestHitCallback = LinkTestHitCallback;
			config.AlignSize = s_AlignSize;
			config.GridSize = s_GridSize;
			s_Context = ed::CreateEditor(&config);
		}
		ed::SetCurrentEditor(s_Context);

		DrawLeftPanel();
		ImGui::SameLine();
		DrawRightPanel();

		ed::SetCurrentEditor(nullptr);
	}

	AnimationGraph* AnimationGraphWindow::GetGraph()
	{
		return m_AnimationGraph.Get();
	}

	void AnimationGraphWindow::DrawLeftPanel()
	{
		const char* createParameterPopupId = "CreateParameterPopup";

		ImGui::BeginChild("Parameters", ImVec2(300, 0));
		ImGui::BeginPaddedArea(ImVec2(5, 5), ImVec2(5, 5));
		auto& parameters = m_AnimationGraph->GetParameters();
		size_t deletedParameter = UINT64_MAX;
		for (size_t i = 0; i < parameters.size(); ++i)
		{
			ImGui::PushID(static_cast<int>(i));
			ImGui::BeginGroup();
			auto& parameter = parameters[i];
			String name = parameter.GetName();
			ImGui::SetNextItemWidth(120);
			if (ImGui::InputText("##name", &name))
			{
				parameter.SetName(name);
			}
			ImGui::SameLine();
			ImGui::SetNextItemWidth(120);
			switch (parameter.GetType())
			{
			case AnimationParameterType::Bool:
			{
				bool value = parameter.GetBoolValue();
				if (ImGui::Checkbox("##value", &value))
				{
					parameter.SetBoolValue(value);
				}
			}
			break;
			case AnimationParameterType::Trigger:
			{
				bool value = parameter.GetTriggerValue();
				if (ImGui::Checkbox("##value", &value))
				{
					parameter.SetTriggerValue(value);
				}
			}
			break;
			case AnimationParameterType::Int:
			{
				int value = parameter.GetIntValue();
				if (ImGui::InputInt("##value", &value, 0))
				{
					parameter.SetIntValue(value);
				}
			}
			break;
			case AnimationParameterType::Float:
			{
				float value = parameter.GetFloatValue();
				if (ImGui::InputFloat("##value", &value))
				{
					parameter.SetFloatValue(value);
				}
			}
			break;
			}
			ImGui::SameLine();
			if (ImGui::Button("X"))
			{
				deletedParameter = i;
			}
			ImGui::EndGroup();
			ImGui::PopID();
		}
		if (deletedParameter != UINT64_MAX)
		{
			parameters.erase(parameters.begin() + deletedParameter);
		}
		if (ImGui::Button("Add Parameter"))
		{
			ImGui::OpenPopup(createParameterPopupId);
		}

		if (ImGui::BeginPopup(createParameterPopupId))
		{
			if (ImGui::MenuItem("Bool"))
			{
				AnimationGraphParameterData parameter = {};
				parameter.SetName("New Bool");
				parameter.SetBoolValue(false);
				parameters.push_back(parameter);
			}
			if (ImGui::MenuItem("Trigger"))
			{
				AnimationGraphParameterData parameter = {};
				parameter.SetName("New Trigger");
				parameter.SetTriggerValue(false);
				parameters.push_back(parameter);
			}
			if (ImGui::MenuItem("Int"))
			{
				AnimationGraphParameterData parameter = {};
				parameter.SetName("New Int");
				parameter.SetIntValue(0);
				parameters.push_back(parameter);
			}
			if (ImGui::MenuItem("Float"))
			{
				AnimationGraphParameterData parameter = {};
				parameter.SetName("New Float");
				parameter.SetFloatValue(0.0f);
				parameters.push_back(parameter);
			}
			ImGui::EndPopup();
		}

		if (ImGui::Button("Save"))
		{
			ImVec2 entryPosition = ed::GetNodePosition(s_EntryNodeId);
			m_StateMachine->SetEntryStatePosition(Vector2(entryPosition.x, entryPosition.y));

			ImVec2 anyPosition = ed::GetNodePosition(s_AnyNodeId);
			m_StateMachine->SetAnyStatePosition(Vector2(anyPosition.x, anyPosition.y));

			for (size_t i = 0; i < m_States.size(); ++i)
			{
				StateData& stateData = m_States[i];
				if (stateData.type == StateType::Default)
				{
					AnimationState* state = static_cast<AnimationState*>(ObjectDB::GetObject(stateData.id));
					ImVec2 position = ed::GetNodePosition(stateData.id);
					state->SetPosition(Vector2(position.x, position.y));
				}
			}
			AssetDB::SetDirty(m_AnimationGraph.Get());
			AssetDB::SaveAssets();
		}
		ImGui::EndPaddedArea();
		ImGui::EndChild();
	}

	void Blueberry::AnimationGraphWindow::DrawRightPanel()
	{
		const char* createStatePopupId = "CreateStatePopup";

		ed::PushStyleVar(ed::StyleVar_LinkStrength, 0.0f);
		ed::PushStyleVar(ed::StyleVar_NodeRounding, 3.0f);
		ed::PushStyleVar(ed::StyleVar_PinRounding, 8.0f);
		ed::PushStyleVar(ed::StyleVar_NodePadding, ImVec4(0, 0, 0, 0));
		ed::PushStyleColor(ed::StyleColor_NodeBg, ImColor(32, 32, 32, 255));
		ed::Begin("AnimationGraph Editor", ImVec2(0.0, 0.0f));

		if (m_IsInitialized)
		{
			for (size_t i = 0; i < m_States.size(); ++i)
			{
				StateData& stateData = m_States[i];
				ObjectId id = stateData.id;

				ed::BeginNode(id);
				ImVec2 nodePos = ed::GetNodePosition(id);

				ImVec2 pos = ImGui::GetCursorPos();
				ImVec2 size = ImVec2(128, 32);
				ImGui::Dummy(size);

				const char* text;
				if (stateData.type == StateType::Default)
				{
					AnimationState* state = static_cast<AnimationState*>(ObjectDB::GetObject(id));
					text = state->GetName().c_str();
				}
				else if (stateData.type == StateType::Entry)
				{
					text = "Entry";
				}
				else if (stateData.type == StateType::Any)
				{
					text = "Any";
				}
				ImVec2 textSize = ImGui::CalcTextSize(text);
				ImVec2 textPos = pos + ImVec2((size.x - textSize.x) * 0.5f, (size.y - textSize.y) * 0.5f);
				ImGui::SetCursorPos(textPos);
				ImGui::Text(text);
				
				ed::BeginPin(stateData.pin, ed::PinKind::Output);
				ed::PinPivotRect(nodePos + size * 0.5f, nodePos + size * 0.5f);
				ed::PinRect(nodePos + size * 0.5f - ImVec2(8, 8), nodePos + size * 0.5f + ImVec2(8, 8));
				ed::EndPin();

				ed::EndNode();
			}

			if (ed::BeginCreate())
			{
				ed::PinId inputPinId, outputPinId;
				if (ed::QueryNewLink(&inputPinId, &outputPinId))
				{
					if (inputPinId && outputPinId)
					{
						size_t fromPin = static_cast<size_t>(inputPinId);
						size_t toPin = static_cast<size_t>(outputPinId);
						StateData& fromStateData = m_States[FindStateIndex(fromPin)];
						StateData& toStateData = m_States[FindStateIndex(toPin)];
						bool isValid = true;
						if (fromPin == toPin || toStateData.type == StateType::Entry || toStateData.type == StateType::Any)
						{
							ed::RejectNewItem();
							isValid = false;
						}
						for (auto& transition : m_Transitions)
						{
							if (transition.fromPin == fromPin && transition.toPin == toPin)
							{
								ed::RejectNewItem();
								isValid = false;
								break;
							}
						}
						if (isValid && ed::AcceptNewItem())
						{
							if (fromStateData.type == StateType::Default)
							{
								AnimationState* fromState = static_cast<AnimationState*>(ObjectDB::GetObject(fromStateData.id));
								AnimationState* toState = static_cast<AnimationState*>(ObjectDB::GetObject(toStateData.id));
								ObjectId transitionId = fromState->CreateTransition(toState)->GetObjectId();
								m_Transitions.push_back({ transitionId, fromPin, toPin });
							}
							else if (fromStateData.type == StateType::Entry)
							{
								AnimationState* toState = static_cast<AnimationState*>(ObjectDB::GetObject(toStateData.id));
								m_StateMachine->SetDefaultState(toState);
								size_t index = FindTransitionIndex(s_EntryNodeId);
								if (index == UINT64_MAX)
								{
									m_Transitions.push_back({ -1, 1, toPin });
								}
								else
								{
									m_Transitions[index].toPin = toPin;
								}
							}
							else if (fromStateData.type == StateType::Any)
							{
								AnimationState* toState = static_cast<AnimationState*>(ObjectDB::GetObject(toStateData.id));
								ObjectId transitionId = m_StateMachine->CreateAnyStateTransition(toState)->GetObjectId();
								m_Transitions.push_back({ transitionId, fromPin, toPin });
							}
						}
					}
				}
			}
			ed::EndCreate();

			if (ed::BeginDelete())
			{
				ed::LinkId linkId = 0;
				while (ed::QueryDeletedLink(&linkId))
				{
					if (ed::AcceptDeletedItem())
					{
						ObjectId transitionId = static_cast<ObjectId>(static_cast<size_t>(linkId));
						for (size_t i = 0; i < m_Transitions.size(); ++i)
						{
							TransitionData& transitionData = m_Transitions[i];
							if (transitionData.id == transitionId)
							{
								StateData& fromState = m_States[FindStateIndex(transitionData.fromPin)];
								if (fromState.type == StateType::Default)
								{
									AnimationTransition* transition = static_cast<AnimationTransition*>(ObjectDB::GetObject(transitionData.id));
									size_t stateIndex = FindStateIndex(transitionData.fromPin);
									AnimationState* state = static_cast<AnimationState*>(ObjectDB::GetObject(m_States[stateIndex].id));
									state->RemoveTransition(transition);
									Object::Destroy(transition);
								}
								else if (fromState.type == StateType::Entry)
								{
									m_StateMachine->SetDefaultState(nullptr);
								}
								else if (fromState.type == StateType::Any)
								{
									AnimationTransition* transition = static_cast<AnimationTransition*>(ObjectDB::GetObject(transitionData.id));
									m_StateMachine->RemoveAnyStateTransition(transition);
									Object::Destroy(transition);
								}
								m_Transitions.erase(m_Transitions.begin() + i);
								break;
							}
						}
					}
				}
				ed::NodeId nodeId = 0;
				while (ed::QueryDeletedNode(&nodeId))
				{
					if (ed::AcceptDeletedItem())
					{
						ObjectId stateId = static_cast<ObjectId>(static_cast<size_t>(nodeId));
						size_t stateIndex = FindStateIndex(stateId);
						StateData& stateData = m_States[stateIndex];
						AnimationState* state = static_cast<AnimationState*>(ObjectDB::GetObject(stateId));
						List<size_t> removedTransitions;
						for (size_t i = 0; i < m_Transitions.size(); ++i)
						{
							TransitionData& transitionData = m_Transitions[i];
							if (transitionData.fromPin == stateData.pin || transitionData.toPin == stateData.pin)
							{
								removedTransitions.push_back(i);
							}
						}

						for (auto it = removedTransitions.rbegin(); it != removedTransitions.rend(); ++it)
						{
							size_t index = *it;
							TransitionData& transitionData = m_Transitions[index];
							StateData& fromStateData = m_States[FindStateIndex(transitionData.fromPin)];
							if (fromStateData.type == StateType::Default)
							{
								AnimationTransition* transition = static_cast<AnimationTransition*>(ObjectDB::GetObject(transitionData.id));
								AnimationState* state = static_cast<AnimationState*>(ObjectDB::GetObject(fromStateData.id));
								state->RemoveTransition(transition);
								Object::Destroy(transition);
							}
							else if (fromStateData.type == StateType::Entry)
							{
								m_StateMachine->SetDefaultState(nullptr);
							}
							else if (fromStateData.type == StateType::Any)
							{
								AnimationTransition* transition = static_cast<AnimationTransition*>(ObjectDB::GetObject(transitionData.id));
								m_StateMachine->RemoveAnyStateTransition(transition);
								Object::Destroy(transition);
							}
							m_Transitions.erase(m_Transitions.begin() + index);
						}
						m_States.erase(m_States.begin() + stateIndex);
						Object::Destroy(state);
					}
				}
			}
			ed::EndDelete();

			for (size_t i = 0; i < m_Transitions.size(); ++i)
			{
				TransitionData& transitionData = m_Transitions[i];
				ed::Link(transitionData.id, transitionData.fromPin, transitionData.toPin);
			}

			ed::Suspend();
			if (ed::ShowBackgroundContextMenu())
			{
				ImGui::OpenPopup(createStatePopupId);
			}

			if (ImGui::BeginPopup(createStatePopupId))
			{
				if (ImGui::MenuItem("State"))
				{
					AnimationState* state = m_StateMachine->CreateState();
					ObjectId id = state->GetObjectId();
					m_States.push_back({ id, StateType::Default, m_MaxPinId++ });
					ImVec2 position = ed::ScreenToCanvas(ImGui::GetMousePos());
					position.x = std::roundf(position.x - std::fmodf(position.x, s_AlignSize));
					position.y = std::roundf(position.y - std::fmodf(position.y, s_AlignSize));
					ed::SetNodePosition(id, position);
				}
				ImGui::EndPopup();
			}
			ed::Resume();
		}
		else
		{
			if (m_AnimationGraph.IsValid())
			{
				m_StateMachine = m_AnimationGraph->GetStateMachine();
				m_States.clear();
				m_Transitions.clear();
				m_MaxPinId = 1;

				m_States.push_back({ s_EntryNodeId, StateType::Entry, m_MaxPinId++ });
				Vector2 entryPosition = m_StateMachine->GetEntryStatePosition();
				ed::SetNodePosition(s_EntryNodeId, ImVec2(entryPosition.x, entryPosition.y));

				m_States.push_back({ s_AnyNodeId, StateType::Any, m_MaxPinId++ });
				Vector2 anyPosition = m_StateMachine->GetAnyStatePosition();
				ed::SetNodePosition(s_AnyNodeId, ImVec2(anyPosition.x, anyPosition.y));

				for (auto& state : m_StateMachine->GetStates())
				{
					ObjectId stateId = state->GetObjectId();
					Vector2 statePosition = state->GetPosition();
					ed::SetNodePosition(stateId, ImVec2(statePosition.x, statePosition.y));
					m_States.push_back({ stateId, StateType::Default, m_MaxPinId++ });
				}
				AnimationState* defaultState = m_StateMachine->GetDefaultState();
				if (defaultState != nullptr)
				{
					m_Transitions.push_back({ s_EntryTransitionId, 1, m_States[FindStateIndex(defaultState->GetObjectId())].pin });
				}
				for (auto& anyStateTransition : m_StateMachine->GetAnyStateTransitions())
				{
					ObjectId transitionId = anyStateTransition.Get()->GetObjectId();
					m_Transitions.push_back({ transitionId, 2, m_States[FindStateIndex(anyStateTransition->GetDestination()->GetObjectId())].pin });
				}
				for (auto& state : m_StateMachine->GetStates())
				{
					ObjectId stateId = state->GetObjectId();
					for (auto& transition : state->GetTransitions())
					{
						ObjectId transitionId = transition.Get()->GetObjectId();
						ObjectId destinationId = transition.Get()->GetDestination()->GetObjectId();

						m_Transitions.push_back({ transitionId, m_States[FindStateIndex(stateId)].pin, m_States[FindStateIndex(destinationId)].pin });
					}
				}
				m_IsInitialized = true;
			}
		}

		ed::End();
		ed::PopStyleVar(4);
		ed::PopStyleColor();

		if (ed::HasSelectionChanged())
		{
			Selection::SetActiveObject(nullptr);
			List<ed::NodeId> selectedNodes(100);
			int selectedNodesCount = ed::GetSelectedNodes(selectedNodes.data(), 100);
			for (int i = 0; i < selectedNodesCount; ++i)
			{
				ObjectId nodeId = static_cast<ObjectId>(static_cast<size_t>(selectedNodes[i]));
				if (nodeId > 0)
				{
					Selection::AddActiveObject(ObjectDB::GetObject(nodeId));
				}
			}
			List<ed::LinkId> selectedLinks(100);
			int selectedLinksCount = ed::GetSelectedLinks(selectedLinks.data(), 100);
			for (int i = 0; i < selectedLinksCount; ++i)
			{
				ObjectId linkId = static_cast<ObjectId>(static_cast<size_t>(selectedLinks[i]));
				if (linkId > 0)
				{
					Selection::AddActiveObject(ObjectDB::GetObject(linkId));
				}
			}
		}
	}

	size_t AnimationGraphWindow::FindStateIndex(size_t pinId)
	{
		for (size_t i = 0; i < m_States.size(); ++i)
		{
			StateData& state = m_States[i];
			if (state.pin == pinId)
			{
				return i;
			}
		}
		return UINT64_MAX;
	}

	size_t AnimationGraphWindow::FindStateIndex(ObjectId id)
	{
		for (size_t i = 0; i < m_States.size(); ++i)
		{
			StateData& state = m_States[i];
			if (state.id == id)
			{
				return i;
			}
		}
		return UINT64_MAX;
	}

	size_t AnimationGraphWindow::FindTransitionIndex(ObjectId id)
	{
		for (size_t i = 0; i < m_Transitions.size(); ++i)
		{
			TransitionData& transition = m_Transitions[i];
			if (transition.id == id)
			{
				return i;
			}
		}
		return UINT64_MAX;
	}

	void AnimationGraphWindow::SetGraph(AnimationGraph* graph)
	{
		m_AnimationGraph = graph;
		m_IsInitialized = false;
	}
}
