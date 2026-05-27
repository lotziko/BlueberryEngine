#include "ImGuiRendererDX12.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "..\DX12\GfxDeviceDX12.h"

#include <imgui\imgui.h>
#include <imgui\imguizmo.h>
#include <imgui\backends\imgui_impl_win32.h>
#include <imgui\backends\imgui_impl_dx12.h>

namespace Blueberry
{
	bool ImGuiRendererDX12::InitializeImpl()
	{
		m_GfxDevice = static_cast<GfxDeviceDX12*>(GfxDevice::GetInstance());

		m_CommandList = m_GfxDevice->GetCommandList();
		m_Hwnd = m_GfxDevice->GetHwnd();

		// Setup Dear ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();

		ImGuiIO* io = &ImGui::GetIO();
		io->IniFilename = NULL;

		ImGui_ImplDX12_InitInfo initInfo = {};
		initInfo.Device = m_GfxDevice->GetDevice();
		initInfo.CommandQueue = m_GfxDevice->GetCommandQueue();
		initInfo.NumFramesInFlight = 2;
		initInfo.RTVFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
		initInfo.DSVFormat = DXGI_FORMAT_D32_FLOAT;
		initInfo.SrvDescriptorHeap = m_GfxDevice->GetCbvSrvDsvRingHeap().GetHeap();
		initInfo.SrvDescriptorAllocFn = [](ImGui_ImplDX12_InitInfo* info, D3D12_CPU_DESCRIPTOR_HANDLE* out_cpu_desc_handle, D3D12_GPU_DESCRIPTOR_HANDLE* out_gpu_desc_handle)
		{
			GfxDeviceDX12* gfxDevice = static_cast<GfxDeviceDX12*>(GfxDevice::GetInstance());
			GfxDescriptorRingHeapDX12& heap = gfxDevice->GetCbvSrvDsvRingHeap();
			GfxRingHandleDX12 handle = heap.AllocatePersistent();
			*out_cpu_desc_handle = handle.GetCPU();
			*out_gpu_desc_handle = handle.GetGPU();
		};
		initInfo.SrvDescriptorFreeFn = [](ImGui_ImplDX12_InitInfo* info, D3D12_CPU_DESCRIPTOR_HANDLE cpu_desc_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_desc_handle)
		{
		};

		// Setup Platform/Renderer backends
		ImGui_ImplWin32_Init(m_Hwnd);
		ImGui_ImplDX12_Init(&initInfo);

		return true;
	}

	void ImGuiRendererDX12::ShutdownImpl()
	{
		// Cleanup
		ImGui_ImplDX12_Shutdown();
		ImGui_ImplWin32_Shutdown();
		ImGui::DestroyContext();
	}

	void ImGuiRendererDX12::BeginImpl()
	{
		// Start the Dear ImGui frame
		ImGui_ImplDX12_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();
		ImGuizmo::BeginFrame();
	}

	void ImGuiRendererDX12::EndImpl()
	{
		// Rendering
		GfxDevice::SetRenderTarget(nullptr);
		ImGui::Render();
		ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), m_CommandList);
		m_GfxDevice->Reset();
	}
}