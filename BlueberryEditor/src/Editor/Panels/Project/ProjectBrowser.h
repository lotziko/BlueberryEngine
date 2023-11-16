#pragma once

#include <filesystem>

namespace Blueberry
{
	class ProjectBrowser
	{
	public:
		ProjectBrowser();

		void DrawUI();

	private:
		std::filesystem::path m_CurrentDirectory;
	};
}