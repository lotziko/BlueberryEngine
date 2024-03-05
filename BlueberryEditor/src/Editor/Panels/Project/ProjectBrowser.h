#pragma once

#include <filesystem>

namespace Blueberry
{
	class ProjectBrowser
	{
	public:
		ProjectBrowser();
		virtual ~ProjectBrowser() = default;

		void DrawUI();

	private:
		std::filesystem::path m_CurrentDirectory;
		const char* m_OpenedModalPopupId = nullptr;
	};
}