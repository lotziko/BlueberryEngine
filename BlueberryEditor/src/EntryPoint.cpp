#include "bbpch.h"
#include "Blueberry\Core\Engine.h"

#include "Editor\EditorLayer.h"

int APIENTRY wWinMain(_In_ HINSTANCE	hInstance,
	_In_opt_ HINSTANCE					hPrevInstance,
	_In_ LPWSTR							lpCmdLine,
	_In_ int							nCmdShow)
{
	AllocConsole();
	BB_INITIALIZE_LOG();

	Blueberry::Engine engine;
	engine.Initialize(Blueberry::WindowProperties("Blueberry Editor", 960, 640, &hInstance));
	engine.PushLayer(new Blueberry::EditorLayer());
	while (engine.ProcessMessages())
	{
		engine.Update();
		engine.Draw();
		Sleep(1000 / 60);
	}
	return 0;
}