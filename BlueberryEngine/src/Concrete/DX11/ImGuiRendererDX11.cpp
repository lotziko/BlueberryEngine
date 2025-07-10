#include "ImGuiRendererDX11.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "..\DX11\GfxDeviceDX11.h"

#include <imgui\imgui.h>
#include <imgui\imguizmo.h>
#include <imgui\backends\imgui_impl_win32.h>
#include <imgui\backends\imgui_impl_dx11.h>

namespace Blueberry
{
	bool ImGuiRendererDX11::InitializeImpl()
	{
		GfxDeviceDX11* gfxDevice = static_cast<GfxDeviceDX11*>(GfxDevice::GetInstance());

		m_Device = gfxDevice->GetDevice();
		m_DeviceContext = gfxDevice->GetDeviceContext();
		m_Hwnd = gfxDevice->GetHwnd();

		// Setup Dear ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();

		ImGuiIO *io = &ImGui::GetIO();
		io->IniFilename = NULL;

		// Setup Platform/Renderer backends
		ImGui_ImplWin32_Init(m_Hwnd);
		ImGui_ImplDX11_Init(m_Device, m_DeviceContext);
		
		return true;
	}

	void ImGuiRendererDX11::ShutdownImpl()
	{
		// Cleanup
		ImGui_ImplDX11_Shutdown();
		ImGui_ImplWin32_Shutdown();
		ImGui::DestroyContext();
	}

	void ImGuiRendererDX11::BeginImpl()
	{
		// Start the Dear ImGui frame
		ImGui_ImplDX11_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();
		ImGuizmo::BeginFrame();
	}

	void ImGuiRendererDX11::EndImpl()
	{
		// Rendering
		ImGui::Render();
		ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
	}
}