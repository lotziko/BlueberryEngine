#pragma once

#include "Editor\Panels\EditorWindow.h"

namespace Blueberry
{
	class LightmappingWindow : public EditorWindow
	{
		OBJECT_DECLARATION(LightmappingWindow)

	public:
		LightmappingWindow() = default;
		virtual ~LightmappingWindow() = default;

		static void Open();

		virtual void OnDrawUI() final;

	private:
		int m_TileSize = 128;
		float m_TexelPerUnit = 5;
		int m_SamplePerTexel = 64;
		int m_PreferredSize = 1024;
		bool m_Denoise = true;
	};
}