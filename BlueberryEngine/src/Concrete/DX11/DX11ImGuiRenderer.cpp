#include "bbpch.h"
#include "DX11ImGuiRenderer.h"

#include "imgui.h"
#include "backends\imgui_impl_win32.h"
#include "backends\imgui_impl_dx11.h"

namespace Blueberry
{
	DX11ImGuiRenderer::DX11ImGuiRenderer(HWND hwnd, ID3D11Device* device, ID3D11DeviceContext* deviceContext) : m_Hwnd(hwnd), m_Device(device), m_DeviceContext(deviceContext)
	{
		// Setup Dear ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();

		//// Setup Platform/Renderer backends
		ImGui_ImplWin32_Init(hwnd);
		ImGui_ImplDX11_Init(device, deviceContext);
	}

	DX11ImGuiRenderer::~DX11ImGuiRenderer()
	{
		// Cleanup
		ImGui_ImplDX11_Shutdown();
		ImGui_ImplWin32_Shutdown();
		ImGui::DestroyContext();
	}

	void DX11ImGuiRenderer::Begin()
	{
		// Start the Dear ImGui frame
		ImGui_ImplDX11_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();
	}

	void DX11ImGuiRenderer::End()
	{
		// Rendering
		ImGui::Render();
		ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
	}
}