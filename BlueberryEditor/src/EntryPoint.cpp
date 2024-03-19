#include "bbpch.h"
#include "Blueberry\Core\Engine.h"
#include "Blueberry\Assets\AssetLoader.h"

#include "Editor\Path.h"
#include "Editor\EditorLayer.h"
#include "Editor\Assets\EditorAssetLoader.h"
#include <chrono>
#include <thread>

int APIENTRY wWinMain(_In_ HINSTANCE	hInstance,
	_In_opt_ HINSTANCE					hPrevInstance,
	_In_ LPWSTR							lpCmdLine,
	_In_ int							nCmdShow)
{
	AllocConsole();
	BB_INITIALIZE_LOG();

	LPWSTR *argList;
	int nArgs = 0;
	argList = CommandLineToArgvW(GetCommandLineW(), &nArgs);

	std::wstring path = std::wstring(argList[1]);
	Blueberry::Path::SetProjectPath(path);

	Blueberry::AssetLoader::Initialize(new Blueberry::EditorAssetLoader());
	Blueberry::Engine engine;
	engine.Initialize(Blueberry::WindowProperties("Blueberry Editor", 960, 640, &hInstance));
	engine.PushLayer(new Blueberry::EditorLayer());

	// Based on https://stackoverflow.com/questions/63429337/limit-fps-in-loop-c
	using framerate = std::chrono::duration<int, std::ratio<1, 60>>;
	auto prev = std::chrono::system_clock::now();
	auto next = prev + framerate{ 1 };
	int N = 0;
	std::chrono::system_clock::duration sum{ 0 };

	while (engine.ProcessMessages())
	{
		std::this_thread::sleep_until(next);
		next += framerate{ 1 };

		engine.Update();
		engine.Draw();

		auto now = std::chrono::system_clock::now();
		sum += now - prev;
		++N;
		prev = now;
	}
	engine.Shutdown();
	return 0;
}