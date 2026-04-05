#include "Blueberry\Core\Application.h"
#include "Blueberry\Core\Window.h"
#include "Blueberry\Logging\Log.h"
#include "Runtime\RuntimeLayer.h"

int APIENTRY wWinMain(_In_ HINSTANCE	hInstance,
	_In_opt_ HINSTANCE					hPrevInstance,
	_In_ LPWSTR							lpCmdLine,
	_In_ int							nCmdShow)
{
	LPWSTR* argList;
	int nArgs = 0;
	argList = CommandLineToArgvW(GetCommandLineW(), &nArgs);

	if (nArgs > 1)
	{
		if (argList[1] == L"DEBUG")
		{
			AllocConsole();
			BB_INITIALIZE_LOG();
		}
	}

	Blueberry::Application application;
	application.Initialize(Blueberry::WindowProperties("Blueberry Runtime", 960, 640, &hInstance, true));
	application.PushLayer(new Blueberry::RuntimeLayer());
	application.Run();
	application.Shutdown();
	return 0;
}