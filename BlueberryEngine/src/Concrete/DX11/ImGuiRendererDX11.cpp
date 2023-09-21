#include "bbpch.h"
#include "ImGuiRendererDX11.h"

#include "imgui.h"
#include "backends\imgui_impl_win32.h"
#include "backends\imgui_impl_dx11.h"

namespace Blueberry
{
	ImGuiRendererDX11::ImGuiRendererDX11(HWND hwnd, ID3D11Device* device, ID3D11DeviceContext* deviceContext) : m_Hwnd(hwnd), m_Device(device), m_DeviceContext(deviceContext)
	{
		// Setup Dear ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();

		//// Setup Platform/Renderer backends
		ImGui_ImplWin32_Init(hwnd);
		ImGui_ImplDX11_Init(device, deviceContext);
	}

	ImGuiRendererDX11::~ImGuiRendererDX11()
	{
		// Cleanup
		ImGui_ImplDX11_Shutdown();
		ImGui_ImplWin32_Shutdown();
		ImGui::DestroyContext();
	}

	void ImGuiRendererDX11::Begin()
	{
		// Start the Dear ImGui frame
		ImGui_ImplDX11_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();
	}

	void ImGuiRendererDX11::End()
	{
		// Rendering
		ImGui::Render();
		ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
	}
}