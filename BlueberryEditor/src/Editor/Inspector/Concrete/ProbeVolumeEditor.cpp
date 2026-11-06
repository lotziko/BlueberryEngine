#include "ProbeVolumeEditor.h"

#include "Blueberry\Scene\Components\ProbeVolume.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Editor\Graphics\ProbeBricksGenerator.h"
#include "Editor\Panels\Scene\SceneArea.h"
#include "Editor\Gizmos\Gizmos.h"
#include "Editor\Misc\ImGuiHelper.h"

namespace Blueberry
{
	static Texture* s_Icon = nullptr;

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

	void ProbeVolumeEditor::OnDrawInspector()
	{
		m_SerializedObject->UpdateIfNeeded();

		if (m_SerializedObject->GetTargets().size() == 1)
		{
			ImGui::Property(&m_SizeProperty, "Size");
			ImGui::Property(&m_SpacingProperty, "Spacing");

			if (ImGui::Button("Bake"))
			{
				ProbeVolume* probeVolume = static_cast<ProbeVolume*>(m_SerializedObject->GetTarget());
				ProbeBricksGenerator::GenerateProbeVolumeBricks(probeVolume);
				SceneArea::RequestRedrawAll();
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
			Vector3 position = transform->GetPosition();
			Vector3 size = probeVolume->GetSize();
			
			if (size.x > 0 && size.y > 0 && size.z > 0)
			{
				Gizmos::SetMatrix(Matrix::Identity);
				Gizmos::DrawBox(position, size);
				if (probeVolume->GetSpacing() > 0)
				{
					Vector3 minPosition = position - size / 2;
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
		}
	}
}
