#include "EditorWindow.h"

#include "Editor\Path.h"
#include "Editor\Serialization\YamlSerializer.h"
#include "Editor\Misc\PlatformHelper.h"
#include "Blueberry\Events\WindowEvents.h"

#include <imgui\imgui.h>
#include <fstream>

namespace Blueberry
{
	OBJECT_DEFINITION(EditorWindow, Object)
	{
		DEFINE_BASE_FIELDS(EditorWindow, Object)
		DEFINE_FIELD(EditorWindow, m_Title, BindingType::String, {})
		DEFINE_FIELD(EditorWindow, m_RawData, BindingType::Raw, FieldOptions().SetSize(37))
	}

	List<ObjectPtr<EditorWindow>> EditorWindow::s_ToRemoveWindows = {};
	List<ObjectPtr<EditorWindow>> EditorWindow::s_ActiveWindows = {};

	static bool s_IsInit = false;

	void EditorWindow::Show()
	{
		Focus();
		for (auto it = s_ActiveWindows.begin(); it < s_ActiveWindows.end(); ++it)
		{
			if (it->IsValid() && it->Get()->m_ObjectId == m_ObjectId)
			{
				return;
			}
		}
		s_ActiveWindows.push_back(this);
	}

	void EditorWindow::ShowPopup()
	{
		m_Flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove;
		Show();
	}

	void EditorWindow::Close()
	{
		s_ToRemoveWindows.push_back(this);
	}

	void EditorWindow::Focus()
	{
		ImGui::SetWindowFocus(m_Title.c_str());
	}

	void EditorWindow::DrawUI()
	{
		bool opened = true;
		bool focused = false;
		ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(100, 100));
		if (ImGui::Begin(m_Title.c_str(), &opened, m_Flags))
		{
			OnDrawUI();
			if (!opened)
			{
				Close();
			}
			focused = true;
		}
		else
		{
			focused = false;
		}
		ImGui::End();
		ImGui::PopStyleVar();
		if (s_IsInit)
		{
			m_Focused = focused;
		}
	}

	void EditorWindow::SetTitle(const String& title)
	{
		m_Title = title;
	}

	const bool& EditorWindow::HasUnsavedChanges()
	{
		return m_HasUnsavedChanges;
	}

	void EditorWindow::SetHasUnsavedChanges(const bool& hasChanges)
	{
		m_HasUnsavedChanges = hasChanges;
		if (hasChanges)
		{
			m_Flags |= ImGuiWindowFlags_UnsavedDocument;
		}
		else
		{
			m_Flags &= ~ImGuiWindowFlags_UnsavedDocument;
		}
	}

	void EditorWindow::Initialize()
	{
		auto layoutPath = Path::GetDataPath();
		layoutPath.append("EditorLayout");

		auto dockPath = Path::GetDataPath();
		dockPath.append("DockLayout");

		if (std::filesystem::exists(layoutPath))
		{
			YamlSerializer serializer = {};
			serializer.Deserialize(layoutPath.string().data());
			for (auto& pair : serializer.GetDeserializedObjects())
			{
				EditorWindow* activeWindow = static_cast<EditorWindow*>(pair.first);
				activeWindow->Show();

				activeWindow->m_Focused = activeWindow->m_RawData[36];
				ImGui::WriteRawWindowData(activeWindow->m_Title.c_str(), reinterpret_cast<char*>(activeWindow->m_RawData));
			}
		}

		if (std::filesystem::exists(dockPath))
		{
			std::ifstream input;
			input.open(dockPath, std::ifstream::binary);

			uint32_t nodeCount;
			input.read(reinterpret_cast<char*>(&nodeCount), sizeof(uint32_t));
			for (uint32_t i = 0; i < nodeCount; ++i)
			{
				char buffer[36];
				input.read(buffer, ARRAYSIZE(buffer));
				ImGui::WriteRawDockNodeData(buffer);
			}
			input.close();
		}
		ImGui::ApplyWindowAndDockNodeData();

		WindowEvents::GetWindowClosing().AddCallback<&EditorWindow::OnWindowClosing>();
	}

	void EditorWindow::Shutdown()
	{
		auto layoutPath = Path::GetDataPath();
		layoutPath.append("EditorLayout");

		auto dockPath = Path::GetDataPath();
		dockPath.append("DockLayout");

		if (s_ActiveWindows.size() > 0)
		{
			ImGui::PrepareWindowData();
			YamlSerializer serializer = {};
			for (auto& activeWindow : s_ActiveWindows)
			{
				const char* title = activeWindow->m_Title.c_str();
				ImGui::ReadRawWindowData(activeWindow->m_Title.c_str(), reinterpret_cast<char*>(activeWindow->m_RawData));
				activeWindow->m_RawData[36] = activeWindow->m_Focused;
				serializer.AddObject(activeWindow.Get());
			}
			serializer.Serialize(layoutPath.string().data());

			ImGui::PrepareDockNodeData();
			uint32_t nodeCount = ImGui::GetDockNodeDataCount();
			if (nodeCount > 0)
			{
				std::ofstream output;
				output.open(dockPath, std::ofstream::binary);
				output.write(reinterpret_cast<char*>(&nodeCount), sizeof(uint32_t));
				for (int i = 0; i < nodeCount; ++i)
				{
					char buffer[36];
					ImGui::ReadRawDockNodeData(i, buffer);
					output.write(buffer, ARRAYSIZE(buffer));
				}
				output.close();
			}
		}

		WindowEvents::GetWindowClosing().RemoveCallback<&EditorWindow::OnWindowClosing>();
	}

	bool EditorWindow::Save(const size_t& type)
	{
		bool result = true;
		for (auto& window : s_ActiveWindows)
		{
			if (window->IsClassType(type))
			{
				if (window->m_HasUnsavedChanges)
				{
					DialogResult dialogResult = PlatformHelper::OpenDialog(L"Unsaved changes", window->GetSaveChangesMessage(), L"Save", L"Don't save", L"Cancel");
					switch (dialogResult)
					{
					case DialogResult::Yes:
						window->OnSaveChanges();
						break;
					case DialogResult::No:
						window->OnDiscardChanges();
						break;
					case DialogResult::Cancel:
						result = false;
						break;
					}
				}
			}
		}
		return result;
	}

	bool EditorWindow::Save()
	{
		return Save(EditorWindow::Type);
	}

	bool EditorWindow::IsFocused(const size_t& type)
	{
		for (auto& window : s_ActiveWindows)
		{
			if (window->IsClassType(type) && window->m_Focused)
			{
				return true;
			}
		}
		return false;
	}

	void EditorWindow::OnWindowClosing(WindowClosingEventArgs& args)
	{
		if (!Save())
		{
			args.Cancel();
		}
	}

	void EditorWindow::Draw()
	{
		if (s_ToRemoveWindows.size() > 0)
		{
			for (auto it = s_ToRemoveWindows.begin(); it < s_ToRemoveWindows.end(); ++it)
			{
				for (auto activeIt = s_ActiveWindows.begin(); activeIt < s_ActiveWindows.end(); ++activeIt)
				{
					if (it->Get()->m_ObjectId == activeIt->Get()->m_ObjectId)
					{
						Object::Destroy(activeIt->Get());
						s_ActiveWindows.erase(activeIt);
						break;
					}
				}
			}
			s_ToRemoveWindows.clear();
		}

		for (auto& window : s_ActiveWindows)
		{
			window->DrawUI();
		}

		if (!s_IsInit)
		{
			s_IsInit = true;
			for (auto& window : s_ActiveWindows)
			{
				if (window->m_Focused)
				{
					ImGui::SetWindowFocus(window->m_Title.c_str());
				}
			}
		}
	}

	EditorWindow* EditorWindow::GetWindow(const size_t& type)
	{
		for (auto& window : s_ActiveWindows)
		{
			if (window->GetType() == type)
			{
				return window.Get();
			}
		}
		const ClassInfo* info = ClassDB::GetInfo(type);
		if (info == nullptr)
		{
			BB_ERROR("Class not exists.");
			return nullptr;
		}
		EditorWindow* window = (EditorWindow*)info->createInstance();
		return window;
	}

	List<ObjectPtr<EditorWindow>>& EditorWindow::GetWindows()
	{
		return s_ActiveWindows;
	}

	void EditorWindow::OnSaveChanges()
	{
	}

	void EditorWindow::OnDiscardChanges()
	{
	}

	WString EditorWindow::GetSaveChangesMessage()
	{
		return L"Window has unsaved changes.";
	}
}
