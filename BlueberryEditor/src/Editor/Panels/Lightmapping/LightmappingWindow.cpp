#include "LightmappingWindow.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Scene\Scene.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Importers\TextureImporter.h"
#include "Editor\Menu\EditorMenuManager.h"
#include "Editor\EditorSceneManager.h"
#include "Editor\Selection.h"
#include "Editor\Misc\TextureHelper.h"
#include "Editor\Scene\SceneSettings.h"
#include "Editor\Scene\LightingData.h"
#include "Editor\Misc\ImGuiHelper.h"

#include <filesystem>
#include <imgui\imgui_internal.h>
#include <directxtex\DirectXTex.h>

namespace Blueberry
{
	OBJECT_DEFINITION(LightmappingWindow, EditorWindow)
	{
		DEFINE_BASE_FIELDS(LightmappingWindow, EditorWindow)
		DEFINE_FIELD(LightmappingWindow, m_TileSize, BindingType::Int, {})
		DEFINE_FIELD(LightmappingWindow, m_TexelPerUnit, BindingType::Float, {})
		DEFINE_FIELD(LightmappingWindow, m_SamplePerTexel, BindingType::Int, {})
		DEFINE_FIELD(LightmappingWindow, m_PreferredSize, BindingType::Int, {})
		DEFINE_FIELD(LightmappingWindow, m_Denoise, BindingType::Bool, {})
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
		Scene* scene = EditorSceneManager::GetScene();
		SceneSettings* settings = EditorSceneManager::GetSettings();
		LightingData* lightingData = settings->GetLightingData();

		if (scene != nullptr)
		{
			ImGui::IntEdit("Tile size", &m_TileSize);
			ImGui::FloatEdit("Texel per unit", &m_TexelPerUnit);
			ImGui::IntEdit("Sample per texel", &m_SamplePerTexel);
			ImGui::IntEdit("Preferred size", &m_PreferredSize);
			ImGui::BoolEdit("Denoise", &m_Denoise);

			switch (LightmappingManager::GetLightmappingState())
			{
				case LightmappingState::None:
				{
					if (ImGui::Button("Bake"))
					{
						try
						{
							CalculationParams params = {};
							params.tileSize = m_TileSize;
							params.texelPerUnit = m_TexelPerUnit;
							params.samplePerTexel = m_SamplePerTexel;
							params.maxSize = m_PreferredSize;
							params.denoise = m_Denoise;

							LightmappingManager::Calculate(EditorSceneManager::GetScene(), params);
						}
						catch (...)
						{
						}
					}

					static float zoom = 1.0f;
					static bool drawAll = false;
					float newZoom = zoom;
					ImGui::SliderFloat("Zoom", &newZoom, 1, 300);
					ImGui::Checkbox("Draw all", &drawAll);

					ImGui::BeginChild("Zoom");

					ImGuiWindow* window = ImGui::GetCurrentWindow();
					window->ScrollTarget = window->Scroll * newZoom / zoom;
					zoom = newZoom;

					ImVec2 pos = ImGui::GetCursorScreenPos();
					ImVec2 size = ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().x);
					size.x *= zoom;
					size.y *= zoom;
					ImDrawList* list = ImGui::GetWindowDrawList();

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
								bool active = Selection::IsActiveObject(meshRenderer->GetEntity());
								if (!active && !drawAll)
								{
									continue;
								}
								ImColor color = active ? ImColor(255, 0, 0) : ImColor(0, 0, 255);//ImColor color = Selection::IsActiveObject(meshRenderer->GetEntity()) ? ImColor(255, 0, 0) : ImColor(0, 255, 0);
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

									list->AddTriangle(pos + ImVec2(p1.x * size.x, p1.y * size.y), pos + ImVec2(p2.x * size.x, p2.y * size.y), pos + ImVec2(p3.x * size.x, p3.y * size.y), color);
								}
							}
						}
						list->PopClipRect();
						ImGui::Dummy(size);
					}
					ImGui::EndChild();
				}
				break;
				case LightmappingState::Calculating:
				{
					ImGui::Text("Calculating...");
					ImGui::ProgressBar(LightmappingManager::GetProgress());
				}
				break;
				case LightmappingState::Waiting:
				{
					CalculationResult& result = LightmappingManager::GetCalculationResult();
					DirectX::ScratchImage image = {};
					image.Initialize2D(DXGI_FORMAT_R32G32B32A32_FLOAT, result.outputSize.x, result.outputSize.y, 1, 1);
					memcpy(image.GetPixels(), result.output.data(), result.outputSize.x * result.outputSize.y * sizeof(Vector4));
					TextureHelper::Flip(image);

					String path = EditorSceneManager::GetPath();
					path.replace(path.find(".scene"), 6, "\\Lightmap.hdr");

					std::filesystem::create_directories(std::filesystem::path(path).parent_path());
					HRESULT hr = DirectX::SaveToHDRFile(*image.GetImages(), WString(path.begin(), path.end()).c_str());
					if (FAILED(hr))
					{
						BB_ERROR("Failed to save texture.");
					}

					auto relativePath = std::filesystem::relative(path, Path::GetAssetsPath());
					TextureImporter* importer = static_cast<TextureImporter*>(AssetDB::GetImporter(relativePath.string().data()));
					importer->SetTextureShape(TextureImporter::TextureShape::Texture2D);
					importer->SetTextureFormat(TextureImporter::TextureFormat::RGBA32);
					importer->SetFilterMode(FilterMode::Trilinear);
					importer->SaveAndReimport();
					AssetDB::Refresh();
					Texture2D* lightmap = static_cast<Texture2D*>(ObjectDB::GetObjectFromGuid(importer->GetGuid(), importer->GetMainObject()));

					if (lightingData == nullptr || lightingData->GetState() != ObjectState::Default)
					{
						lightingData = Object::Create<LightingData>();
						settings->SetLightingData(lightingData);
					}
					lightingData->SetLightmap(lightmap);
					lightingData->SetChartScaleOffset(result.chartOffsetScale);
					lightingData->SetInstanceOffset(result.chartInstanceOffset);
					lightingData->Apply(scene);

					path = EditorSceneManager::GetPath();
					path.replace(path.find(".scene"), 6, "\\LightingData.asset");
					relativePath = std::filesystem::relative(path, Path::GetAssetsPath());
					AssetDB::CreateAsset(lightingData, relativePath.string().data());
					EditorSceneManager::Save();
					LightmappingManager::Shutdown();
				}
				break;
			}
		}
	}
}