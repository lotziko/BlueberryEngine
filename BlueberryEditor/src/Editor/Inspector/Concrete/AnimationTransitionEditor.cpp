#include "AnimationTransitionEditor.h"

#include "Blueberry\Animations\AnimationGraph.h"
#include "Editor\Panels\AnimationGraph\AnimationGraphWindow.h"
#include "Editor\Misc\ImGuiHelper.h"

namespace Blueberry
{
	AnimationGraph* s_Graph;

	void AnimationTransitionEditor::OnEnable()
	{
		m_IsFixedDurationProperty = m_SerializedObject->FindProperty("m_IsFixedDuration");
		m_TransitionDurationProperty = m_SerializedObject->FindProperty("m_TransitionDuration");
		m_TransitionOffsetProperty = m_SerializedObject->FindProperty("m_TransitionOffset");
		m_HasExitTimeProperty = m_SerializedObject->FindProperty("m_HasExitTime");
		m_ExitTimeProperty = m_SerializedObject->FindProperty("m_ExitTime");
		m_ConditionsProperty = m_SerializedObject->FindProperty("m_Conditions");
		AnimationGraphWindow* window = static_cast<AnimationGraphWindow*>(EditorWindow::GetWindow(AnimationGraphWindow::Type));
		if (window != nullptr)
		{
			s_Graph = window->GetGraph();
		}
	}

	void AnimationTransitionEditor::OnDrawInspector()
	{
		ImGui::Property(&m_IsFixedDurationProperty);
		ImGui::Property(&m_TransitionDurationProperty);
		ImGui::Property(&m_TransitionOffsetProperty);
		ImGui::Property(&m_HasExitTimeProperty);
		if (m_HasExitTimeProperty.GetBool())
		{
			ImGui::Property(&m_ExitTimeProperty);
		}

		auto& parameters = s_Graph->GetParameters();
		for (size_t i = 0; i < m_ConditionsProperty.GetListSize(); ++i)
		{
			SerializedProperty conditionProperty = m_ConditionsProperty.GetListElement(i);
			SerializedProperty nameProperty = conditionProperty.FindProperty("m_Name");
			String name = nameProperty.GetString();
			ImGui::PushID(static_cast<int>(i));
			float width = ImGui::GetContentRegionAvail().x;
			ImGui::SetNextItemWidth(width * 0.5f);
			if (ImGui::BeginCombo("##name", name.c_str()))
			{
				for (auto& parameter : parameters)
				{
					const String& parameterName = parameter.GetName();
					if (ImGui::Selectable(parameterName.c_str()))
					{
						nameProperty.SetString(parameterName);
					}
				}
				ImGui::EndCombo();
			}
			ImGui::SameLine();
			for (auto& parameter : parameters)
			{
				if (parameter.GetName() == name)
				{
					AnimationParameterType parameterType = parameter.GetType();
					SerializedProperty comparisonProperty = conditionProperty.FindProperty("m_Comparison");
					SerializedProperty valueProperty = conditionProperty.FindProperty("m_Value");
					switch (parameterType)
					{
					case AnimationParameterType::Bool:
					{
						bool value = valueProperty.GetFloat() > 0.0f;
						if (ImGui::Checkbox("##value", &value))
						{
							valueProperty.SetFloat(value ? 1.0f : 0.0f);
						}
					}
					break;
					case AnimationParameterType::Int:
					{
						static List<std::pair<String, int>> intComparisons =
						{
							std::make_pair("Greater", static_cast<int>(AnimationConditionComparison::Greater)),
							std::make_pair("Less", static_cast<int>(AnimationConditionComparison::Less)),
							std::make_pair("Equal", static_cast<int>(AnimationConditionComparison::Equal)),
							std::make_pair("NotEqual", static_cast<int>(AnimationConditionComparison::NotEqual))
						};

						int comparison = static_cast<int>(comparisonProperty.GetInt());
						ImGui::SetNextItemWidth(width * 0.2f);
						if (ImGui::InputEnum("##comparison", &comparison, &intComparisons))
						{
							comparisonProperty.SetInt(comparison);
						}
						ImGui::SameLine();

						int value = static_cast<int>(valueProperty.GetFloat());
						ImGui::SetNextItemWidth(width * 0.2f);
						if (ImGui::InputInt("##value", &value, 0))
						{
							valueProperty.SetFloat(static_cast<float>(value));
						}
					}
					break;
					case AnimationParameterType::Float:
					{
						static List<std::pair<String, int>> floatComparisons =
						{
							std::make_pair("Greater", static_cast<int>(AnimationConditionComparison::Greater)),
							std::make_pair("Less", static_cast<int>(AnimationConditionComparison::Less))
						};

						int comparison = static_cast<int>(comparisonProperty.GetInt());
						ImGui::SetNextItemWidth(width * 0.2f);
						if (ImGui::InputEnum("##comparison", &comparison, &floatComparisons))
						{
							comparisonProperty.SetInt(comparison);
						}
						ImGui::SameLine();

						float value = valueProperty.GetFloat();
						ImGui::SetNextItemWidth(width * 0.2f);
						if (ImGui::InputFloat("##value", &value))
						{
							valueProperty.SetFloat(value);
						}
					}
					break;
					}
					break;
				}
			}
			ImGui::SameLine();
			if (ImGui::Button("X"))
			{
				m_ConditionsProperty.DeleteListElement(i);
			}
			ImGui::PopID();
		}
		if (ImGui::Button("Add Condition"))
		{
			size_t index = m_ConditionsProperty.GetListSize();
			m_ConditionsProperty.InsertListElement(index);
			m_ConditionsProperty.GetListElement(index).FindProperty("m_Name").SetString(parameters.size() > 0 ? parameters[0].GetName() : "");
		}
		m_SerializedObject->ApplyModifiedProperties();
	}
}