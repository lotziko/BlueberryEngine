#pragma once

#include "Blueberry\Graphics\ImGuiRenderer.h"

namespace Blueberry
{
	class ImGuiRendererDX11 final : public ImGuiRenderer
	{
	public:
		virtual bool InitializeImpl() final;
		virtual void ShutdownImpl() final;

	protected:
		virtual void BeginImpl() final;
		virtual void EndImpl() final;

	private:
		HWND m_Hwnd;
		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;
	};
}