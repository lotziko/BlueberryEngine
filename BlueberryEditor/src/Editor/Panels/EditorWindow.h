#pragma once

#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Core\Structs.h"

namespace Blueberry
{
	class WindowClosingEventArgs;

	class EditorWindow : public Object
	{
		OBJECT_DECLARATION(EditorWindow)

	public:
		EditorWindow() = default;
		virtual ~EditorWindow() = default;

		void Show();
		void ShowPopup();
		void Close();
		void Focus();

		void DrawUI();

		void SetTitle(const String& title);

		const bool& HasUnsavedChanges();
		void SetHasUnsavedChanges(const bool& hasChanges);

		static void Initialize();
		static void Shutdown();
		static bool Save(const size_t& type);
		static bool Save();
		static bool IsFocused(const size_t& type);

		static void OnWindowClosing(WindowClosingEventArgs& args);

		static void Draw();
		static EditorWindow* GetWindow(const size_t& type);
		static List<ObjectPtr<EditorWindow>>& GetWindows();

	protected:
		virtual void OnDrawUI() = 0;
		virtual void OnSaveChanges();
		virtual void OnDiscardChanges();
		virtual WString GetSaveChangesMessage();

	protected:
		String m_Title;
		uint8_t m_RawData[37];

	private:
		static List<ObjectPtr<EditorWindow>> s_ToRemoveWindows;
		static List<ObjectPtr<EditorWindow>> s_ActiveWindows;
		bool m_Focused = false;
		bool m_HasUnsavedChanges = false;
		int m_Flags = 0;
	};
}