#include "MeshEditor.h"

#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\GfxRenderTexturePool.h"

#include "Editor\Preview\MeshPreview.h"
#include "Editor\Misc\ImGuiHelper.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	MeshEditor::MeshEditor()
	{
		m_RenderTexture = GfxRenderTexturePool::Get(512, 512, 1);
	}

	MeshEditor::~MeshEditor()
	{
		GfxRenderTexturePool::Release(m_RenderTexture);
	}

	void DrawUVChannel(Mesh* mesh, const uint32_t& channel, const ImVec2& pos, const ImVec2& size)
	{
		ImDrawList* list = ImGui::GetWindowDrawList();
		ImColor backgroundColor = ImColor(0, 0, 0);
		static ImColor lineColors[] = { ImColor(255, 255, 255), ImColor(0, 255, 255), ImColor(255, 0, 255), ImColor(255, 255, 0), ImColor(0, 0, 255), ImColor(255, 0, 0), ImColor(0, 255, 0) };
		//ImColor lineColor = ImColor(255, 255, 255);
		uint32_t* indices = mesh->GetIndices();
		float* uvs = mesh->GetUVs(channel);
		uint32_t uvSize = mesh->GetUVSize(channel);

		list->AddRectFilled(pos, pos + size, backgroundColor);
		list->PushClipRect(pos, pos + size, true);
		if (uvs != nullptr)
		{
			for (uint32_t i = 0; i < mesh->GetIndexCount(); i += 3)
			{
				Vector3 p1 = *reinterpret_cast<Vector3*>(uvs + indices[i] * uvSize);
				Vector3 p2 = *reinterpret_cast<Vector3*>(uvs + indices[i + 1] * uvSize);
				Vector3 p3 = *reinterpret_cast<Vector3*>(uvs + indices[i + 2] * uvSize);
				list->AddTriangle(pos + ImVec2(p1.x * size.x, p1.y * size.y), pos + ImVec2(p2.x * size.x, p2.y * size.y), pos + ImVec2(p3.x * size.x, p3.y * size.y), /*lineColor*/lineColors[static_cast<uint32_t>(p3.z) % ARRAYSIZE(lineColors)]);
			}
		}
		list->PopClipRect();
		ImGui::Dummy(size);
	}

	void MeshEditor::OnDrawInspector()
	{
		Mesh* mesh = static_cast<Mesh*>(m_SerializedObject->GetTarget());

		static int previewType;
		static List<String> previewTypes = { "Default", "UV 0", "UV 1" };
		ImGui::Text(mesh->GetName().c_str());
		ImGui::Text(std::to_string(mesh->GetVertexCount()).c_str());
		ImGui::Text(std::to_string(mesh->GetIndexCount()).c_str());
		ImGui::EnumEdit("Preview", &previewType, &previewTypes);

		ImVec2 pos = ImGui::GetCursorScreenPos();
		ImVec2 size = ImGui::GetContentRegionAvail();
		ImVec2 imageSize = ImVec2(size.x, (m_RenderTexture->GetHeight() * size.x) / static_cast<float>(m_RenderTexture->GetWidth()));

		switch (previewType)
		{
		case 0:
		{
			static MeshPreview preview;
			preview.Draw(mesh, m_RenderTexture);
			ImGui::Image(reinterpret_cast<ImTextureID>(m_RenderTexture->GetHandle()), imageSize, ImVec2(0, 0), ImVec2(1, 1));
		}
		break;
		case 1:
		{
			DrawUVChannel(mesh, 0, pos, imageSize);
		}
		break;
		case 2:
		{
			DrawUVChannel(mesh, 1, pos, imageSize);
		}
		}
	}
}