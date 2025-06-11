#pragma once

#include "Editor\Panels\EditorWindow.h"

namespace Blueberry
{
	class GfxTexture;

	class GameView : public EditorWindow
	{
		OBJECT_DECLARATION(GameView)

	public:
		GameView() = default;
		virtual ~GameView() = default;

		static void Open();

		virtual void OnDrawUI();

	private:
		GfxTexture* m_RenderTarget = nullptr;
	};
}