#pragma once

#include "Blueberry\Graphics\ImGuiRenderer.h"

class DX11ImGuiRenderer final : public ImGuiRenderer
{
public:
	DX11ImGuiRenderer(HWND hwnd, ID3D11Device* device, ID3D11DeviceContext* deviceContext);
	virtual ~DX11ImGuiRenderer() final;

	virtual void Begin() final;
	virtual void End() final;
private:
	HWND m_Hwnd;
	ID3D11Device* m_Device;
	ID3D11DeviceContext* m_DeviceContext;
};