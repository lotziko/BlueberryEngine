#pragma once

#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Core\Structs.h"

namespace Blueberry
{
	class EditorWindow : public Object
	{
		OBJECT_DECLARATION(EditorWindow)

	public:
		EditorWindow() = default;
		virtual ~EditorWindow() = default;

		void Show();
		void ShowPopup();
		void Close();

		void DrawUI();

		void SetTitle(const std::string& title);

		static void Load();
		static void Save();

		static void Draw();
		static EditorWindow* GetWindow(const size_t& type);
		static List<ObjectPtr<EditorWindow>>& GetWindows();

	protected:
		virtual void OnDrawUI() = 0;

	protected:
		std::string m_Title;
		uint8_t m_RawData[37];

	private:
		static List<ObjectPtr<EditorWindow>> s_ToRemoveWindows;
		static List<ObjectPtr<EditorWindow>> s_ActiveWindows;
		bool m_Focused = false;
		int m_Flags = 0;
	};
}