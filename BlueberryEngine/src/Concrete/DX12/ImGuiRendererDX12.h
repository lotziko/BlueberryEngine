#pragma once

#include "Blueberry\Graphics\ImGuiRenderer.h"
#include "Concrete\DX12\DX12.h"

namespace Blueberry
{
	class GfxDeviceDX12;

	class ImGuiRendererDX12 final : public ImGuiRenderer
	{
	public:
		virtual bool InitializeImpl() final;
		virtual void ShutdownImpl() final;

	protected:
		virtual void BeginImpl() final;
		virtual void EndImpl() final;

	private:
		HWND m_Hwnd;
		GfxDeviceDX12* m_GfxDevice;
		ID3D12GraphicsCommandList* m_CommandList;
	};
}