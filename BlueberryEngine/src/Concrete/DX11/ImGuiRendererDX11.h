#pragma once

#include "Blueberry\Graphics\ImGuiRenderer.h"

namespace Blueberry
{
	class ImGuiRendererDX11 final : public ImGuiRenderer
	{
	public:
		ImGuiRendererDX11(HWND hwnd, ID3D11Device* device, ID3D11DeviceContext* deviceContext);
		virtual ~ImGuiRendererDX11() final;

		virtual void Begin() final;
		virtual void End() final;
	private:
		HWND m_Hwnd;
		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;
	};
}