#include "bbpch.h"
#include "ProjectBrowser.h"

#include "Editor\Path.h"
#include "Editor\Assets\AssetImporter.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\EditorSceneManager.h"
#include "Editor\Selection.h"
#include "Editor\Prefabs\PrefabManager.h"
#include "Editor\Panels\Inspector\InspectorExpandedItemsCache.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Component.h"
#include "imgui\imgui_internal.h"

#include "Editor\Assets\ThumbnailCache.h"
#include "Editor\Assets\IconDB.h"

namespace Blueberry
{
	const int bottomPanelSize = 20;
	const int cellSize = 90;
	const int spaceBetweenCells = 15;
	const int cellIconPadding = 8;

	ProjectBrowser::ProjectBrowser()
	{
		m_CurrentDirectory = Path::GetAssetsPath();

		m_FolderIconSmall = (Texture2D*)AssetLoader::Load("assets/icons/FolderIconSmall.png");
		m_FolderIconSmallOpened = (Texture2D*)AssetLoader::Load("assets/icons/FolderIconSmallOpened.png");
		
		UpdateTree();
		AssetDB::GetAssetDBRefreshed().AddCallback<ProjectBrowser, &ProjectBrowser::OnAssetDBRefresh>(this);
	}

	ProjectBrowser::~ProjectBrowser()
	{
		AssetDB::GetAssetDBRefreshed().RemoveCallback<ProjectBrowser, &ProjectBrowser::OnAssetDBRefresh>(this);
	}

	void ProjectBrowser::DrawUI()
	{
		ImGui::Begin("Project");

		ImGuiTableFlags tableFlags = 0;
		tableFlags |= ImGuiTableFlags_Resizable;

		if (ImGui::BeginTable("##ProjectTable", 2, tableFlags))
		{
			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex(0);

			ImGui::BeginChild("##FoldersColumn", ImVec2(0, ImGui::GetContentRegionAvail().y - 2));
			DrawFoldersTree();
			ImGui::EndChild();

			ImGui::TableSetColumnIndex(1);
			ImGui::BeginChild("##CurrentFolderColumn", ImVec2(0, ImGui::GetContentRegionAvail().y - 2));
			DrawCurrentFolder();
			ImGui::EndChild();

			ImGui::EndTable();
		}
		
		ImGui::End();
	}

	void ProjectBrowser::DrawFoldersTree()
	{
		DrawFolderNode(m_FolderTree.GetRoot());
	}

	void ProjectBrowser::DrawFolderNode(const FolderTreeNode& node)
	{
		ImGuiTreeNodeFlags flags = (node.children.size() > 0 ? ImGuiTreeNodeFlags_OpenOnArrow : ImGuiTreeNodeFlags_Leaf) | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_SpanFullWidth;
		ImVec2 pos = ImGui::GetCursorScreenPos();
		const char* label = node.name.c_str();
		bool opened = ImGui::TreeNodeEx(label, flags, "");

		if (!ImGui::IsItemToggledOpen() && ImGui::IsItemClicked())
		{
			m_CurrentDirectory = node.path;
		}

		ImGui::SameLine();
		ImGui::SetCursorPosX(ImGui::GetCursorPosX() - 4);
		ImGui::Image((opened && node.children.size() > 0) ? m_FolderIconSmallOpened->GetHandle() : m_FolderIconSmall->GetHandle(), ImVec2(16, 16), ImVec2(0, 1), ImVec2(1, 0));
		ImGui::SameLine();
		ImGui::SetCursorPosX(ImGui::GetCursorPosX() - 4);
		ImGui::Text("%s", label);

		if (opened)
		{
			for (auto& child : node.children)
			{
				DrawFolderNode(child);
			}
			ImGui::TreePop();
		}
	}

	void ProjectBrowser::DrawCurrentFolder()
	{
		ImVec2 pos = ImGui::GetCursorScreenPos();
		ImVec2 size = ImGui::GetContentRegionAvail();
		bool isAnyFileHovered = false;

		if (m_CurrentDirectory != m_PreviousDirectory)
		{
			UpdateFiles();
		}

		if (ImGui::BeginChild("Middle panel", ImVec2(size.x, size.y - bottomPanelSize)))
		{
			ImVec2 panelPos = ImGui::GetCursorPos();
			int maxCells = (int)floorf(size.x / (cellSize + spaceBetweenCells));
			if (maxCells > 0)
			{
				float expandedSpaceBetweenCells = (size.x - (maxCells * cellSize)) / (maxCells + 1);
				// Calculate expected size and items positions
				Vector2 expectedCursorPos = Vector2::Zero;
				uint32_t cellIndex = 0;
				for (auto& asset : m_CurrentDirectoryAssets)
				{
					asset.expanded = InspectorExpandedItemsCache::Get(asset.pathString);
					for (int i = 0, n = asset.expanded ? asset.objects.size() : 1; i < n; ++i)
					{
						expectedCursorPos.x += expandedSpaceBetweenCells;
						asset.positions[i] = Vector2(expectedCursorPos.x, expectedCursorPos.y);
						if (cellIndex + 1 < maxCells)
						{
							expectedCursorPos.x += cellSize;
							++cellIndex;
						}
						else
						{
							expectedCursorPos.y += cellSize + spaceBetweenCells;
							expectedCursorPos.x = 0;
							cellIndex = 0;
						}
					}
				}
				ImGui::Dummy(ImVec2(size.x, expectedCursorPos.y + cellSize));
				ImGui::SetCursorPos(panelPos);

				// Prefab creating
				if (ImGui::BeginDragDropTarget())
				{
					const ImGuiPayload* payload = ImGui::GetDragDropPayload();
					if (payload != nullptr && payload->IsDataType("OBJECT_ID"))
					{
						Blueberry::ObjectId* id = (Blueberry::ObjectId*)payload->Data;
						Blueberry::Object* object = Blueberry::ObjectDB::GetObject(*id);
						if (object != nullptr && object->IsClassType(Entity::Type) && ImGui::AcceptDragDropPayload("OBJECT_ID"))
						{
							PrefabManager::CreatePrefab(m_CurrentDirectory.string(), (Entity*)object);
							UpdateFiles();
						}
					}
					ImGui::EndDragDropTarget();
				}

				// Draw items
				float scrollY = ImGui::GetScrollY();
				float topClip = scrollY - cellSize - spaceBetweenCells;
				float bottomClip = scrollY + ImGui::GetWindowHeight();
				for (auto& asset : m_CurrentDirectoryAssets)
				{
					for (int i = 0, n = asset.expanded ? asset.objects.size() : 1; i < n; ++i)
					{
						Object* object;
						if (i == 0)
						{
							object = asset.importer;
						}
						else
						{
							object = asset.objects[i];
						}
						Vector2 position = asset.positions[i];
						if (position.y >= topClip && position.y <= bottomClip)
						{
							ImGui::SetCursorPos(ImVec2(position.x, position.y));
							// Expanded background
							if (i > 0 && asset.expanded)
							{
								ImVec2 screenPos = ImGui::GetCursorScreenPos();
								ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(screenPos.x - 12, screenPos.y), ImVec2(screenPos.x + cellSize + 12, screenPos.y + cellSize), ImGui::GetColorU32(ImGuiCol_TitleBg), 3.0f);
							}
							DrawObject(object, asset, isAnyFileHovered);
						}
					}
					// Expand icon
					if (asset.objects.size() > 1)
					{
						const int iconSize = 16;
						Vector2 position = asset.positions[0];
						ImGui::SetCursorPos(ImVec2(position.x + cellSize, position.y + cellSize / 2 - iconSize / 2));
						ImGui::SetItemAllowOverlap();
						ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
						if (ImGui::ArrowButtonEx("##Arrow", asset.expanded ? ImGuiDir_Left : ImGuiDir_Right, ImVec2(iconSize, iconSize)))
						{
							asset.expanded = !asset.expanded;
							InspectorExpandedItemsCache::Set(asset.pathString, asset.expanded);
						}
						ImGui::PopStyleColor();
					}
				}
			}

			if (!isAnyFileHovered && ImGui::IsWindowHovered() && ImGui::IsMouseClicked(0))
			{
				Selection::SetActiveObject(nullptr);
			}
		}
		ImGui::EndChild();

		ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::GetStyleColorVec4(ImGuiCol_MenuBarBg));
		ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0.0f);
		if (ImGui::BeginChild("Bottom panel"))
		{
			std::string path = AssetDB::GetRelativeAssetPath(Selection::GetActiveObject());
			if (path.size() > 0)
			{
				ImGui::Text(path.c_str());
			}
		}
		ImGui::EndChild();
		ImGui::PopStyleVar();
		ImGui::PopStyleColor();

		const char* popId = "Delete?";
		const char* popupId = "ProjectPopup";
		if (ImGui::BeginPopup(popupId))
		{
			// TODO refactor
			if (ImGui::MenuItem("Scene"))
			{
				EditorSceneManager::CreateEmpty("");
			}
			if (ImGui::MenuItem("Material"))
			{
				m_OpenedModalPopupId = popId;
			}

			ImGui::EndPopup();
		}

		if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(1))
		{
			ImGui::OpenPopup(popupId);
		}

		if (m_OpenedModalPopupId == popId)
		{
			ImGui::OpenPopup(popId);
		}
		if (ImGui::BeginPopupModal(popId, NULL, ImGuiWindowFlags_AlwaysAutoResize))
		{
			static char name[256];
			ImGui::InputText("Name", name, 256);
			ImGui::Separator();

			if (ImGui::Button("OK", ImVec2(120, 0)))
			{
				std::string materialName(name);
				materialName.append(".material");

				auto relativePath = std::filesystem::relative(m_CurrentDirectory, Path::GetAssetsPath());
				relativePath.append(materialName);

				Material* material = Object::Create<Material>();
				AssetDB::CreateAsset(material, relativePath.string());
				AssetDB::SaveAssets();
				AssetDB::Refresh();

				m_OpenedModalPopupId = nullptr;
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndPopup();
		}

		if (ImGui::IsKeyPressed(ImGuiKey_S) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
		{
			AssetDB::SaveAssets();
		}
	}

	void ProjectBrowser::DrawObject(Object* object, const AssetInfo& asset, bool& anyHovered)
	{
		ImVec2 screenPos = ImGui::GetCursorScreenPos();

		ImGui::PushID(object->GetObjectId());
		ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, ImVec2(0.5f, 1.0f));
		
		if (ImGui::Selectable(object->GetName().c_str(), Selection::IsActiveObject(object), 0, ImVec2(cellSize, cellSize)))
		{
			if (ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
			{
				Selection::AddActiveObject(object);
			}
			else
			{
				Selection::SetActiveObject(object);
			}
			asset.importer->ImportDataIfNeeded();
		}

		if (ImGui::IsItemHovered())
		{
			if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
			{
				OpenAsset(asset);
			}
			anyHovered = true;
		}

		Object* iconObject = object;
		if (object == asset.importer && asset.objects.size() > 0 && asset.objects[0] != nullptr)
		{
			iconObject = asset.objects[0];
		}

		// Dragging
		if (ImGui::BeginDragDropSource())
		{
			ObjectId objectId = iconObject->GetObjectId();
			ImGui::SetDragDropPayload("OBJECT_ID", &objectId, sizeof(ObjectId));
			ImGui::Text("%s", iconObject->GetName().c_str());
			ImGui::EndDragDropSource();
		}

		Texture* icon = ThumbnailCache::GetThumbnail(iconObject);
		if (icon == nullptr)
		{
			icon = IconDB::GetAssetIcon(iconObject);
		}
		ImGui::GetWindowDrawList()->AddImage(icon->GetHandle(), ImVec2(screenPos.x + cellIconPadding, screenPos.y), ImVec2(screenPos.x + cellSize - cellIconPadding, screenPos.y + cellSize - cellIconPadding * 2), ImVec2(0, 1), ImVec2(1, 0));

		ImGui::PopStyleVar();
		ImGui::PopID();
	}

	void ProjectBrowser::OpenAsset(const AssetInfo& asset)
	{
		if (asset.isDirectory)
		{
			std::filesystem::path directoryPath = asset.importer->GetFilePath();
			m_CurrentDirectory /= directoryPath.filename();
		}
		else
		{
			std::string stringPath = asset.importer->GetFilePath();
			std::filesystem::path filePath = stringPath;
			if (filePath.extension() == ".scene")
			{
				EditorSceneManager::Load(stringPath);
			}
			else
			{
				// TODO
			}
		}
	}

	void ProjectBrowser::OnAssetDBRefresh()
	{
		UpdateTree();
		UpdateFiles();
	}

	void ProjectBrowser::UpdateTree()
	{
		m_FolderTree.Update(Path::GetAssetsPath().string());
	}

	void ProjectBrowser::UpdateFiles()
	{
		m_PreviousDirectory = m_CurrentDirectory;
		m_CurrentDirectoryAssets.clear();

		for (auto& it : std::filesystem::directory_iterator(m_CurrentDirectory))
		{
			auto path = it.path();
			if (it.is_directory() && path.extension() != ".meta")
			{
				auto relativePath = std::filesystem::relative(path, Path::GetAssetsPath());
				AssetInfo info = {};
				info.path = path;
				info.pathString = path.string();
				info.importer = AssetDB::GetImporter(relativePath.string());
				info.objects.emplace_back(nullptr);
				info.positions.resize(1);
				info.isDirectory = true;
				m_CurrentDirectoryAssets.emplace_back(info);
			}
		}
		for (auto& it : std::filesystem::directory_iterator(m_CurrentDirectory))
		{
			auto path = it.path();

			if (!it.is_directory() && path.extension() != ".meta")
			{
				auto relativePath = std::filesystem::relative(path, Path::GetAssetsPath());
				AssetImporter* importer = AssetDB::GetImporter(relativePath.string());
				if (importer != nullptr)
				{
					Object* mainObject = ObjectDB::GetObjectFromGuid(importer->GetGuid(), importer->GetMainObject());
					AssetInfo info = {};
					info.path = relativePath.string();
					info.importer = importer;
					if (mainObject != nullptr)
					{
						info.objects.emplace_back(mainObject);
					}
					for (auto& pair : importer->GetAssetObjects())
					{
						Object* object = ObjectDB::GetObject(pair.second);
						if (object != nullptr)
						{
							info.objects.emplace_back(object);
						}
					}
					if (info.objects.size() == 0)
					{
						info.objects.emplace_back(nullptr);
					}
					info.positions.resize(Max(1, info.objects.size()));
					info.isDirectory = false;
					m_CurrentDirectoryAssets.emplace_back(info);
				}
			}
		}
	}
}
