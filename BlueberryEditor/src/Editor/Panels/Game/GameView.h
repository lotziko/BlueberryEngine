#pragma once

namespace Blueberry
{
	class RenderTexture;

	class GameView
	{
	public:
		GameView();
		~GameView();

		void DrawUI();

	private:
		RenderTexture* m_RenderTarget = nullptr;
	};
}