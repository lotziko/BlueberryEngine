#include "Blueberry\Core\Application.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Core\Window.h"
#include "Blueberry\Core\EngineLayer.h"
#include "Blueberry\Logging\Log.h"

#include "Editor\Path.h"
#include "Editor\HubLayer.h"
#include "Editor\EditorLayer.h"
#include "Editor\Assets\EditorAssetLoader.h"

#include "Blueberry\Core\ClassDB.h"

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

	Blueberry::AssetLoader::Initialize(new Blueberry::EditorAssetLoader());
	Blueberry::Application application;
	application.Initialize(Blueberry::WindowProperties("Blueberry Editor", 960, 640, &hInstance, true));
	if (nArgs > 1)
	{
		Blueberry::WString path = Blueberry::WString(argList[1]);
		Blueberry::Path::SetProjectPath(path);
		application.PushLayer(new Blueberry::EngineLayer());
		application.PushLayer(new Blueberry::EditorLayer());
	}
	else
	{
		application.PushLayer(new Blueberry::HubLayer([&application](Blueberry::Layer* caller, Blueberry::WString path)
		{
			Blueberry::Path::SetProjectPath(path);
			application.PopLayer(caller);
			application.PushLayer(new Blueberry::EngineLayer());
			application.PushLayer(new Blueberry::EditorLayer());
		}));
	}
	application.Run();
	application.Shutdown();
	return 0;
}