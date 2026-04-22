#include "RuntimeLayer.h"

#include "Blueberry\Core\Time.h"
#include "Blueberry\Core\Timer.h"
#include "Blueberry\Core\Screen.h"
#include "Blueberry\Core\Application.h"
#include "Blueberry\Core\Window.h"
#include "Blueberry\Core\EngineLayer.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexturePool.h"
#include "Blueberry\Graphics\ImGuiRenderer.h"
#include "Blueberry\Graphics\RmlUiRenderer.h"
#include "Blueberry\Events\WindowEvents.h"
#include "Blueberry\Physics\Physics.h"
#include "Blueberry\Physics\PhysicsShapeCache.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Input\Cursor.h"
#include "Blueberry\Input\Input.h"
#include "Blueberry\Audio\Audio.h"

#include "Runtime\RuntimeLoader.h"
#include "Runtime\Assets\RuntimeAssetLoader.h"
#include "Runtime\Physics\RuntimePhysicsShapeCache.h"
#include "Runtime\Graphics\GameViewRenderer.h"

namespace Blueberry
{
	void RuntimeLayer::OnAttach()
	{
		m_Scene = new Scene();
		Application::SetRunning(true);
		EngineLayer::Register();
		Physics::Initialize();
		Physics::Enable();
		Audio::Initialize();
		AssetLoader::Initialize(new RuntimeAssetLoader());
		RuntimeLoader::LoadAssets();
		PhysicsShapeCache::Initialize(new RuntimePhysicsShapeCache());
		RmlUiRenderer::Initialize();
		RuntimeLoader::LoadScene(m_Scene);
		EngineLayer::Initialize();
		ImGuiRenderer::Initialize();
		Input::SetEnabled(true);

		Window* window = Application::GetInstance()->GetWindow();
		Screen::SetGameViewport(Rectangle(0, 0, window->GetWidth(), window->GetHeight()));
		WindowEvents::GetWindowResized().AddCallback<RuntimeLayer, &RuntimeLayer::OnWindowResize>(this);
	}

	void RuntimeLayer::OnDetach()
	{
		Physics::Disable();
		Physics::Shutdown();
		Audio::Shutdown();
		RmlUiRenderer::Shutdown();
		GfxTexturePool::Shutdown();
		EngineLayer::Shutdown();
		ImGuiRenderer::Shutdown();
		Application::SetRunning(false);

		WindowEvents::GetWindowResized().RemoveCallback<RuntimeLayer, &RuntimeLayer::OnWindowResize>(this);
	}

	void RuntimeLayer::OnUpdate()
	{
		m_Scene->FixedUpdate();
		Physics::Update(Time::GetFixedDeltaTime());
		Audio::Update();
	}

	void RuntimeLayer::OnDraw()
	{
		if (Input::IsKeyPressed(KeyCode::MouseLeft))
		{
			Screen::SetAllowCursorLock(true);
		}

		if (Input::IsKeyPressed(KeyCode::Escape))
		{
			Screen::SetAllowCursorLock(false);
		}
		
		Application::GetInstance()->GetWindow()->SetCursor(!(Screen::IsAllowCursorLock() && Cursor::IsHidden()));

		m_Scene->Update();
		GfxDevice::ClearColor({ 0, 0, 0, 1 });
		
		GameViewRenderer::Draw(m_Scene);
		GfxDevice::SwapBuffers();

		Time::EndFrame();
		Timer::Update();
		GfxTexturePool::Update();
		EngineLayer::Update();
	}

	void RuntimeLayer::OnWindowResize(const WindowResizeEventArgs& args)
	{
		GfxDevice::ResizeBackbuffer(args.GetWidth(), args.GetHeight());
		Screen::SetGameViewport(Rectangle(0l, 0l, static_cast<long>(args.GetWidth()), static_cast<long>(args.GetHeight())));
	}
}