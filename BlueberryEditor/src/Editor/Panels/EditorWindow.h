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

		void SetTitle(const String& title);
		void SetMaximized(const bool& maximized);

		const bool& HasUnsavedChanges();
		void SetHasUnsavedChanges(const bool& hasChanges);

		static void Initialize();
		static void Shutdown();
		static bool Save(const TypeId& type);
		static bool Save();
		static bool IsFocused(const TypeId& type);

		static void OnWindowClosing(WindowClosingEventArgs& args);

		static void Draw();
		static EditorWindow* GetWindow(const TypeId& type);
		static List<ObjectPtr<EditorWindow>>& GetWindows();

	protected:
		virtual void OnDrawUI() = 0;
		virtual void OnSaveChanges();
		virtual void OnDiscardChanges();
		virtual WString GetSaveChangesMessage();

	private:
		void DrawUI();
		void DrawMaximizedUI();

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