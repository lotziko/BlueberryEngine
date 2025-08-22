#include "LightmappingWindow.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\LightingData.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Importers\TextureImporter.h"
#include "Editor\Menu\EditorMenuManager.h"
#include "Editor\EditorSceneManager.h"
#include "Editor\Selection.h"
#include "Editor\Misc\TextureHelper.h"

#include "Baking\LightmappingManager.h"

#include <filesystem>
#include <imgui\imgui_internal.h>
#include <directxtex\DirectXTex.h>

namespace Blueberry
{
	OBJECT_DEFINITION(LightmappingWindow, EditorWindow)
	{
		DEFINE_BASE_FIELDS(LightmappingWindow, EditorWindow)
		EditorMenuManager::AddItem("Window/Lightmapping", &LightmappingWindow::Open);
	}

	void LightmappingWindow::Open()
	{
		EditorWindow* window = GetWindow(LightmappingWindow::Type);
		window->SetTitle("Lightmapping");
		window->Show();
	}

	void LightmappingWindow::OnDrawUI()
	{
		const uint32_t tileSize = 128;
		const uint32_t texelPerUnit = 4;
		Scene* scene = EditorSceneManager::GetScene();
		if (scene != nullptr)
		{
			if (ImGui::Button("Bake"))
			{
				try
				{
					uint8_t* result;
					Vector2Int size;
					List<Vector4> scaleOffset;
					Dictionary<ObjectId, uint32_t> instanceOffset;
					LightmappingManager::Calculate(EditorSceneManager::GetScene(), Vector2Int(tileSize, tileSize), result, size, scaleOffset, instanceOffset);
					
					DirectX::ScratchImage image = {};
					image.Initialize2D(DXGI_FORMAT_R32G32B32A32_FLOAT, size.x, size.y, 1, 1);
					memcpy(image.GetPixels(), result, size.x * size.y * sizeof(Vector4));
					TextureHelper::Flip(image);

					String path = EditorSceneManager::GetPath();
					path.replace(path.find(".scene"), 6, "\\Lightmap.hdr");

					std::filesystem::create_directories(std::filesystem::path(path).parent_path());
					HRESULT hr = DirectX::SaveToHDRFile(*image.GetImages(), WString(path.begin(), path.end()).c_str());
					if (FAILED(hr))
					{
						BB_ERROR("Failed to save texture.");
					}

					AssetDB::Refresh();
					auto relativePath = std::filesystem::relative(path, Path::GetAssetsPath());
					TextureImporter* importer = static_cast<TextureImporter*>(AssetDB::GetImporter(relativePath.string().data()));
					importer->SetTextureShape(TextureImporter::TextureShape::Texture2D);
					importer->SetTextureFormat(TextureImporter::TextureFormat::BC6H);
					importer->SetFilterMode(FilterMode::Trilinear);
					importer->SaveAndReimport();
					Texture2D* lightmap = static_cast<Texture2D*>(ObjectDB::GetObjectFromGuid(importer->GetGuid(), importer->GetMainObject()));

					LightingData* lightingData = scene->GetSettings()->GetLightingData();
					if (lightingData == nullptr || lightingData->GetState() != ObjectState::Default)
					{
						lightingData = Object::Create<LightingData>();
						scene->GetSettings()->SetLightingData(lightingData);
					}
					lightingData->SetLightmap(lightmap);
					lightingData->SetChartScaleOffset(scaleOffset);
					lightingData->SetInstanceOffset(instanceOffset);
					lightingData->Apply(scene);

					path = EditorSceneManager::GetPath();
					path.replace(path.find(".scene"), 6, "\\LightingData.asset");
					relativePath = std::filesystem::relative(path, Path::GetAssetsPath());
					AssetDB::CreateAsset(lightingData, relativePath.string().data());
					EditorSceneManager::Save();
					BB_FREE(result);
				}
				catch (...)
				{
				}
			}

			static float zoom = 1.0f;
			float newZoom = zoom;
			ImGui::SliderFloat("Zoom", &newZoom, 1, 300);

			ImGui::BeginChild("Zoom");

			ImGuiWindow* window = ImGui::GetCurrentWindow();
			window->ScrollTarget = window->Scroll * newZoom / zoom;
			zoom = newZoom;

			ImVec2 pos = ImGui::GetCursorScreenPos();
			ImVec2 size = ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().x);
			size.x *= zoom;
			size.y *= zoom;
			ImDrawList* list = ImGui::GetWindowDrawList();

			LightingData* lightingData = scene->GetSettings()->GetLightingData();
			if (lightingData != nullptr && lightingData->GetState() == ObjectState::Default)
			{
				Texture2D* lightmap = lightingData->GetLightmap();
				if (lightmap != nullptr)
				{
					list->AddImage(reinterpret_cast<ImTextureID>(lightmap->GetHandle()), pos, pos + size);
				}

				list->PushClipRect(pos, pos + size, true);
				Vector4* chartScaleOffset = lightingData->GetChartScaleOffset();
				for (auto& component : scene->GetIterator<MeshRenderer>())
				{
					MeshRenderer* meshRenderer = static_cast<MeshRenderer*>(component.second);
					if (meshRenderer->GetLightmapChartOffset() != 0)
					{
						if (!Selection::IsActiveObject(meshRenderer->GetEntity()))
						{
							continue;
						}
						//ImColor color = Selection::IsActiveObject(meshRenderer->GetEntity()) ? ImColor(255, 0, 0) : ImColor(0, 255, 0);
						Mesh* mesh = meshRenderer->GetMesh();
						uint32_t* indices = mesh->GetIndices();
						Vector3* uvs = reinterpret_cast<Vector3*>(mesh->GetUVs(1));
						uint32_t indexCount = mesh->GetIndexCount();

						for (uint32_t i = 0; i < indexCount; i += 3)
						{
							Vector3 p1 = uvs[indices[i]];
							Vector3 p2 = uvs[indices[i + 1]];
							Vector3 p3 = uvs[indices[i + 2]];

							Vector4 scaleOffset = chartScaleOffset[static_cast<uint32_t>(meshRenderer->GetLightmapChartOffset() + p3.z)];

							p1.x = p1.x * scaleOffset.z + scaleOffset.x;
							p2.x = p2.x * scaleOffset.z + scaleOffset.x;
							p3.x = p3.x * scaleOffset.z + scaleOffset.x;

							p1.y = p1.y * scaleOffset.w + scaleOffset.y;
							p2.y = p2.y * scaleOffset.w + scaleOffset.y;
							p3.y = p3.y * scaleOffset.w + scaleOffset.y;

							list->AddTriangle(pos + ImVec2(p1.x * size.x, p1.y * size.y), pos + ImVec2(p2.x * size.x, p2.y * size.y), pos + ImVec2(p3.x * size.x, p3.y * size.y), ImColor(255, 0, 0));
						}
					}
				}
				list->PopClipRect();
				ImGui::Dummy(size);
			}
			ImGui::EndChild();
		}
	}
}