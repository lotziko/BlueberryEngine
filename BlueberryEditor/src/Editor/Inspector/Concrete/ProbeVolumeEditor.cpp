#include "ProbeVolumeEditor.h"

#include "Blueberry\Scene\Components\ProbeVolume.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Editor\Graphics\ProbeBricksGenerator.h"
#include "Editor\Panels\Scene\SceneArea.h"
#include "Editor\Gizmos\Gizmos.h"
#include "Editor\Misc\ImGuiHelper.h"
#include "Editor\Undo\UndoManager.h"

#include <imgui\imgui.h>
#include <imgui\imguizmo.h>

namespace Blueberry
{
	static Texture* s_Icon = nullptr;
	static bool s_ShowBricks = false;
	static bool s_EditBounds = false;

	ProbeVolumeEditor::ProbeVolumeEditor()
	{
		if (s_Icon == nullptr)
		{
			s_Icon = static_cast<Texture*>(AssetLoader::Load("assets/icons/ProbeVolumeIcon.png"));
		}
	}

	void ProbeVolumeEditor::OnEnable()
	{
		m_SizeProperty = m_SerializedObject->FindProperty("m_Size");
		m_SpacingProperty = m_SerializedObject->FindProperty("m_Spacing");
	}

	void ProbeVolumeEditor::OnDisable()
	{
		s_EditBounds = false;
	}

	void ProbeVolumeEditor::OnDrawInspector()
	{
		m_SerializedObject->UpdateIfNeeded();

		if (s_ShowBricks)
		{
			if (ImGui::Button("Hide bricks"))
			{
				s_ShowBricks = false;
			}
		}
		else
		{
			if (ImGui::Button("Show bricks"))
			{
				s_ShowBricks = true;
			}
		}

		if (m_SerializedObject->GetTargetCount() == 1)
		{
			if (s_EditBounds)
			{
				if (ImGui::Button("Stop edit bounds"))
				{
					s_EditBounds = false;
				}
			}
			else
			{
				if (ImGui::Button("Start edit bounds"))
				{
					s_EditBounds = true;
				}
			}
		}

		ImGui::Property(&m_SizeProperty, "Size");
		ImGui::Property(&m_SpacingProperty, "Spacing");

		if (ImGui::Button("Bake"))
		{
			for (Object* target : m_SerializedObject->GetTargets())
			{
				ProbeVolume* probeVolume = static_cast<ProbeVolume*>(target);
				ProbeBricksGenerator::GenerateProbeVolumeBricks(probeVolume);
			}
		}

		m_SerializedObject->ApplyModifiedProperties();
	}

	Texture* ProbeVolumeEditor::GetIcon(Object* object)
	{
		return s_Icon;
	}

	void ProbeVolumeEditor::OnDrawSceneSelected()
	{
		for (Object* target : m_SerializedObject->GetTargets())
		{
			ProbeVolume* probeVolume = static_cast<ProbeVolume*>(target);
			Transform* transform = probeVolume->GetTransform();
			Vector3 position = Vector3::Zero;
			Vector3 size = probeVolume->GetSize();
			
			Gizmos::SetMatrix(transform->GetLocalToWorldMatrix());
			Gizmos::DrawBox(position, size);

			if (size.x > 0 && size.y > 0 && size.z > 0)
			{
				if (s_ShowBricks && probeVolume->GetSpacing() > 0)
				{
					Vector3 minPosition = -size / 2;
					Vector3Int density = probeVolume->GetDensity();
					Vector3 cellSize = Vector3(size.x / density.x, size.y / density.y, size.z / density.z);
					for (int i = 0; i < density.x; ++i)
					{
						for (int j = 0; j < density.y; ++j)
						{
							for (int k = 0; k < density.z; ++k)
							{
								Gizmos::DrawBox(minPosition + cellSize * Vector3(i + 0.5f, j + 0.5f, k + 0.5f), cellSize);
							}
						}
					}
				}
			}

			if (s_EditBounds)
			{
				if (Gizmos::BoxBoundsHandle(position, size))
				{
					UndoManager::RecordObject(probeVolume);
					UndoManager::RecordObject(transform);
					transform->SetPosition(transform->GetPosition() + Math::MultiplyVector(transform->GetLocalToWorldMatrix(), position));
					probeVolume->SetSize(size);
				}
			}
		}
	}
}
