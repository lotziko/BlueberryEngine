#include "MeshInspector.h"

#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\RenderTexture.h"

#include "Editor\Preview\MeshPreview.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	MeshInspector::MeshInspector()
	{
		m_RenderTexture = RenderTexture::Create(512, 512, 1);
	}

	MeshInspector::~MeshInspector()
	{
		Object::Destroy(m_RenderTexture);
	}

	void MeshInspector::Draw(Object* object)
	{
		static MeshPreview preview;
		Mesh* mesh = static_cast<Mesh*>(object);
		preview.Draw(mesh, m_RenderTexture);

		ImVec2 size = ImGui::GetContentRegionAvail();
		ImGui::Image(reinterpret_cast<ImTextureID>(m_RenderTexture->GetHandle()), ImVec2(size.x, (m_RenderTexture->GetHeight() * size.x) / static_cast<float>(m_RenderTexture->GetWidth())), ImVec2(0, 1), ImVec2(1, 0));
	}
}