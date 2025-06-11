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

#include "Editor\Assets\ThumbnailCache.h"
#include "Editor\Assets\IconDB.h"
#include "Editor\Menu\EditorMenuManager.h"
#include "Editor\Misc\ImGuiHelper.h"

#include <imgui\imgui_internal.h>

namespace Blueberry
{
	OBJECT_DEFINITION(ProjectBrowser, EditorWindow)
	{
		DEFINE_BASE_FIELDS(ProjectBrowser, EditorWindow)
		DEFINE_FIELD(ProjectBrowser, m_FoldersColumnWidth, BindingType::Float, {})
		EditorMenuManager::AddItem("Window/Project", &ProjectBrowser::Open);
	}

	ProjectBrowser::ProjectBrowser()
	{
		m_CurrentDirectory = Path::GetAssetsPath();

		m_FolderIconSmall = static_cast<Texture2D*>(AssetLoader::Load("assets/icons/FolderIconSmall.png"));
		m_FolderIconSmallOpened = static_cast<Texture2D*>(AssetLoader::Load("assets/icons/FolderIconSmallOpened.png"));
		
		UpdateTree();
		AssetDB::GetAssetDBRefreshed().AddCallback<ProjectBrowser, &ProjectBrowser::OnAssetDBRefresh>(this);
	}

	ProjectBrowser::~ProjectBrowser()
	{
		AssetDB::GetAssetDBRefreshed().RemoveCallback<ProjectBrowser, &ProjectBrowser::OnAssetDBRefresh>(this);
	}

	void ProjectBrowser::Open()
	{
		EditorWindow* window = GetWindow(ProjectBrowser::Type);
		window->SetTitle("Project");
		window->Show();
	}

	bool wasInit = false;

	void ProjectBrowser::OnDrawUI()
	{
		ImGuiIO& io = ImGui::GetIO();
		ImVec2 pos = ImGui::GetCursorPos();

		ImGui::BeginChild("##FoldersColumn", ImVec2(m_FoldersColumnWidth, ImGui::GetContentRegionAvail().y - 2));
		DrawFoldersTree();
		ImGui::EndChild();

		ImGui::SameLine();
		ImGui::HorizontalSplitter("##FolderHorizontalSplitter", &m_FoldersColumnWidth, 200);

		ImGui::SameLine();
		ImGui::BeginChild("##CurrentFolderColumn", ImVec2(0, ImGui::GetContentRegionAvail().y - 2));
		DrawCurrentFolder();
		ImGui::EndChild();
	}

	void ProjectBrowser::DrawFoldersTree()
	{
		DrawFolderNode(m_FolderTree.GetRoot());
	}

	void ProjectBrowser::DrawFolderNode(const FolderTreeNode& node)
	{
		ImGui::EditorStyle& style = ImGui::GetEditorStyle();

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
		ImGui::Image(reinterpret_cast<ImTextureID>((opened && node.children.size() > 0) ? m_FolderIconSmallOpened->GetHandle() : m_FolderIconSmall->GetHandle()), ImVec2(style.ProjectFolderIconSize, style.ProjectFolderIconSize), ImVec2(0, 1), ImVec2(1, 0));
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
		ImGui::EditorStyle& style = ImGui::GetEditorStyle();

		ImVec2 pos = ImGui::GetCursorScreenPos();
		ImVec2 size = ImGui::GetContentRegionAvail();
		bool isAnyFileHovered = false;

		if (m_CurrentDirectory != m_PreviousDirectory)
		{
			UpdateFiles();
		}

		if (ImGui::BeginChild("Middle panel", ImVec2(size.x, size.y - style.ProjectBottomPanelSize)))
		{
			ImVec2 panelPos = ImGui::GetCursorPos();
			uint32_t maxCells = static_cast<uint32_t>(floorf(size.x / (style.ProjectCellSize + style.ProjectSpaceBetweenCells)));
			if (maxCells > 0)
			{
				float expandedSpaceBetweenCells = (size.x - (maxCells * style.ProjectCellSize)) / (maxCells + 1);
				// Calculate expected size and items positions
				Vector2 expectedCursorPos = Vector2::Zero;
				uint32_t cellIndex = 0;
				for (auto& asset : m_CurrentDirectoryAssets)
				{
					asset.expanded = InspectorExpandedItemsCache::Get(asset.pathString);
					for (uint32_t i = 0, n = asset.expanded ? static_cast<uint32_t>(asset.objects.size()) : 1; i < n; ++i)
					{
						expectedCursorPos.x += expandedSpaceBetweenCells;
						asset.positions[i] = Vector2(expectedCursorPos.x, expectedCursorPos.y);
						if (cellIndex + 1 < maxCells)
						{
							expectedCursorPos.x += style.ProjectCellSize;
							++cellIndex;
						}
						else
						{
							expectedCursorPos.y += style.ProjectCellSize + style.ProjectSpaceBetweenCells;
							expectedCursorPos.x = 0;
							cellIndex = 0;
						}
					}
				}
				ImGui::Dummy(ImVec2(size.x, expectedCursorPos.y + style.ProjectCellSize));
				ImGui::SetCursorPos(panelPos);

				// Prefab creating
				if (ImGui::BeginDragDropTarget())
				{
					const ImGuiPayload* payload = ImGui::GetDragDropPayload();
					if (payload != nullptr && payload->IsDataType("OBJECT_ID"))
					{
						Blueberry::ObjectId* id = static_cast<Blueberry::ObjectId*>(payload->Data);
						Blueberry::Object* object = Blueberry::ObjectDB::GetObject(*id);
						if (object != nullptr && object->IsClassType(Entity::Type) && ImGui::AcceptDragDropPayload("OBJECT_ID"))
						{
							PrefabManager::CreatePrefab(m_CurrentDirectory.string().data(), static_cast<Entity*>(object));
							UpdateFiles();
						}
					}
					ImGui::EndDragDropTarget();
				}

				// Draw items
				float scrollY = ImGui::GetScrollY();
				float topClip = scrollY - style.ProjectCellSize - style.ProjectSpaceBetweenCells;
				float bottomClip = scrollY + ImGui::GetWindowHeight();
				for (auto& asset : m_CurrentDirectoryAssets)
				{
					for (uint32_t i = 0, n = asset.expanded ? static_cast<uint32_t>(asset.objects.size()) : 1; i < n; ++i)
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
								ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(screenPos.x - 12, screenPos.y), ImVec2(screenPos.x + style.ProjectCellSize + 12, screenPos.y + style.ProjectCellSize), ImGui::GetColorU32(ImGuiCol_TitleBg), 3.0f);
							}
							DrawObject(object, asset, isAnyFileHovered);
						}
					}
					// Expand icon
					if (asset.objects.size() > 1)
					{
						Vector2 position = asset.positions[0];
						ImGui::SetCursorPos(ImVec2(position.x + style.ProjectCellSize, position.y + style.ProjectCellSize / 2 - style.ProjectExpandIconSize / 2));
						ImGui::SetItemAllowOverlap();
						ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
						ImGui::PushID(asset.objects[0]->GetObjectId());
						if (ImGui::ArrowButtonEx("##Arrow", asset.expanded ? ImGuiDir_Left : ImGuiDir_Right, ImVec2(style.ProjectExpandIconSize, style.ProjectExpandIconSize)))
						{
							asset.expanded = !asset.expanded;
							InspectorExpandedItemsCache::Set(asset.pathString, asset.expanded);
						}
						ImGui::PopID();
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
			String path = AssetDB::GetRelativeAssetPath(Selection::GetActiveObject());
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
				AssetDB::CreateAsset(material, relativePath.string().data());
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
		ImGui::EditorStyle& style = ImGui::GetEditorStyle();

		ImVec2 screenPos = ImGui::GetCursorScreenPos();

		ImGui::PushID(object->GetObjectId());
		ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, ImVec2(0.5f, 1.0f));
		
		if (ImGui::Selectable(object->GetName().c_str(), Selection::IsActiveObject(object), 0, ImVec2(style.ProjectCellSize, style.ProjectCellSize)))
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
		ImGui::GetWindowDrawList()->AddImage(reinterpret_cast<ImTextureID>(icon->GetHandle()), ImVec2(screenPos.x + style.ProjectCellIconPadding, screenPos.y), ImVec2(screenPos.x + style.ProjectCellSize - style.ProjectCellIconPadding, screenPos.y + style.ProjectCellSize - style.ProjectCellIconPadding * 2), ImVec2(0, 1), ImVec2(1, 0));

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
			String stringPath = asset.importer->GetFilePath();
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
		m_FolderTree.Update(Path::GetAssetsPath().string().data());
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
				info.importer = AssetDB::GetImporter(relativePath.string().data());
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
				AssetImporter* importer = AssetDB::GetImporter(relativePath.string().data());
				if (importer != nullptr)
				{
					Object* mainObject = ObjectDB::GetObjectFromGuid(importer->GetGuid(), importer->GetMainObject());
					AssetInfo info = {};
					info.path = relativePath;
					info.pathString = relativePath.string();
					info.importer = importer;
					if (mainObject != nullptr)
					{
						info.objects.emplace_back(mainObject);
					}
					for (auto& pair : importer->GetAssetObjects())
					{
						Object* object = ObjectDB::GetObject(pair.second);
						if (object != nullptr && object != mainObject)
						{
							info.objects.emplace_back(object);
						}
					}
					if (info.objects.size() == 0)
					{
						info.objects.emplace_back(nullptr);
					}
					info.positions.resize(std::max(1ull, info.objects.size()));
					info.isDirectory = false;
					m_CurrentDirectoryAssets.emplace_back(info);
				}
			}
		}
	}
}
