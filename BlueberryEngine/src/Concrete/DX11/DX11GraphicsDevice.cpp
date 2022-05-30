#include "bbpch.h"
#include "DX11GraphicsDevice.h"
#include "DX11Shader.h"
#include "DX11Buffer.h"
#include "DX11Texture.h"
#include "DX11ImGuiRenderer.h"

DX11GraphicsDevice::DX11GraphicsDevice()
{
}

bool DX11GraphicsDevice::Initialize(int width, int height, void* data)
{
	if (!InitializeDirectX(*(static_cast<HWND*>(data)), width, height))
		return false;

	return true;
}

void DX11GraphicsDevice::ClearColor(const Color& color) const
{
	m_DeviceContext->ClearRenderTargetView(m_RenderTargetView.Get(), color);
}

void DX11GraphicsDevice::SwapBuffers() const
{
	m_SwapChain->Present(1, NULL);
}

void DX11GraphicsDevice::SetViewport(int x, int y, int width, int height)
{
	D3D11_VIEWPORT viewport;
	ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));

	viewport.TopLeftX = x;
	viewport.TopLeftY = y;
	viewport.Width = static_cast<FLOAT>(width);
	viewport.Height = static_cast<FLOAT>(height);

	m_DeviceContext->RSSetViewports(1, &viewport);
}

void DX11GraphicsDevice::ResizeBackbuffer(int width, int height)
{
	m_DeviceContext->OMSetRenderTargets(0, 0, 0);
	m_RenderTargetView->Release();

	HRESULT hr;

	hr = m_SwapChain->ResizeBuffers(1, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, 0);
	if (FAILED(hr))
	{
		BB_ERROR(WindowsHelper::GetErrorMessage(hr, "ResizeBuffers failed."));
		return;
	}
	
	ID3D11Texture2D* backBuffer;
	hr = m_SwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&backBuffer));
	if (FAILED(hr))
	{
		BB_ERROR(WindowsHelper::GetErrorMessage(hr, "GetBuffer failed."));
		return;
	}

	hr = m_Device->CreateRenderTargetView(backBuffer, NULL, m_RenderTargetView.GetAddressOf());
	if (FAILED(hr))
	{
		BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create render target view."));
		return;
	}

	backBuffer->Release();

	m_DeviceContext->OMSetRenderTargets(1, m_RenderTargetView.GetAddressOf(), NULL);

	D3D11_VIEWPORT viewport;
	ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));

	viewport.TopLeftX = 0;
	viewport.TopLeftY = 0;
	viewport.Width = static_cast<FLOAT>(width);
	viewport.Height = static_cast<FLOAT>(height);

	m_DeviceContext->RSSetViewports(1, &viewport);
}

bool DX11GraphicsDevice::CreateShader(const std::wstring& vertexShaderPath, const std::wstring& pixelShaderPath, Ref<Shader>& shader)
{
	auto dxShader = CreateRef<DX11Shader>(m_Device.Get(), m_DeviceContext.Get());
	if (!dxShader->Initialize(vertexShaderPath, pixelShaderPath))
	{
		return false;
	}
	shader = dxShader;
	return true;
}

bool DX11GraphicsDevice::CreateVertexBuffer(const VertexLayout& layout, const UINT& vertexCount, Ref<VertexBuffer>& buffer)
{
	auto dxBuffer = CreateRef<DX11VertexBuffer>(m_Device.Get(), m_DeviceContext.Get());
	if (!dxBuffer->Initialize(layout, vertexCount))
	{
		return false;
	}
	buffer = dxBuffer;
	return true;
}

bool DX11GraphicsDevice::CreateIndexBuffer(const UINT& indexCount, Ref<IndexBuffer>& buffer)
{
	auto dxBuffer = CreateRef<DX11IndexBuffer>(m_Device.Get(), m_DeviceContext.Get());
	if (!dxBuffer->Initialize(indexCount))
	{
		return false;
	}
	buffer = dxBuffer;
	return true;
}

bool DX11GraphicsDevice::CreateConstantBuffer(const UINT& byteSize, Ref<ConstantBuffer>& buffer)
{
	auto dxBuffer = CreateRef<DX11ConstantBuffer>(m_Device.Get(), m_DeviceContext.Get());
	if (!dxBuffer->Initialize(byteSize))
	{
		return false;
	}
	buffer = dxBuffer;
	return true;
}

bool DX11GraphicsDevice::CreateTexture(const std::string& path, Ref<Texture>& texture) const
{
	auto dxTexture = CreateRef<DX11Texture>(m_Device.Get(), m_DeviceContext.Get());
	if (!dxTexture->Initialize(path))
	{
		return false;
	}
	texture = dxTexture;
	return true;
}

bool DX11GraphicsDevice::CreateImGuiRenderer(Ref<ImGuiRenderer>& renderer) const
{
	auto dxRenderer = CreateRef<DX11ImGuiRenderer>(m_Hwnd, m_Device.Get(), m_DeviceContext.Get());
	renderer = dxRenderer;
	return true;
}

void DX11GraphicsDevice::Draw(const int& vertices) const
{
	m_DeviceContext->RSSetState(m_RasterizerState.Get());
	m_DeviceContext->OMSetDepthStencilState(m_DepthStencilState.Get(), 0);
	m_DeviceContext->Draw(vertices, 0);
}

void DX11GraphicsDevice::DrawIndexed(const int& indices) const
{
	m_DeviceContext->RSSetState(m_RasterizerState.Get());
	m_DeviceContext->OMSetDepthStencilState(m_DepthStencilState.Get(), 0);
	m_DeviceContext->DrawIndexed(indices, 0, 0);
}

Matrix DX11GraphicsDevice::GetGPUMatrix(const Matrix& viewProjection) const
{
	Matrix copy;
	viewProjection.Transpose(copy);
	return copy;
}

bool DX11GraphicsDevice::InitializeDirectX(HWND hwnd, int width, int height)
{
	m_Hwnd = hwnd;

	DXGI_SWAP_CHAIN_DESC scd;
	ZeroMemory(&scd, sizeof(DXGI_SWAP_CHAIN_DESC));

	scd.BufferDesc.Width = width;
	scd.BufferDesc.Height = height;
	scd.BufferDesc.RefreshRate.Numerator = 60;
	scd.BufferDesc.RefreshRate.Denominator = 1;
	scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	scd.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
	scd.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;

	scd.SampleDesc.Count = 1;
	scd.SampleDesc.Quality = 0;

	scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	scd.BufferCount = 1;
	scd.OutputWindow = hwnd;
	scd.Windowed = TRUE;
	scd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
	scd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

	HRESULT hr;
	hr = D3D11CreateDeviceAndSwapChain(
		NULL,
		D3D_DRIVER_TYPE_HARDWARE, //hardware driver
		NULL, //software driver
		D3D11_CREATE_DEVICE_DEBUG, //no flags
		NULL, //feature levels
		0, //no feature levels
		D3D11_SDK_VERSION,
		&scd, //swapchain description
		m_SwapChain.GetAddressOf(), //m_SwapChain address
		m_Device.GetAddressOf(), //m_Device address
		NULL, //supported feature level
		m_DeviceContext.GetAddressOf() //m_DeviceContext address
	);

	if (FAILED(hr))
	{
		BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating swapchain."));
		return false;
	}

	ID3D11Texture2D* backBuffer;
	hr = m_SwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&backBuffer));
	if (FAILED(hr))
	{
		BB_ERROR(WindowsHelper::GetErrorMessage(hr, "GetBuffer failed."));
		return false;
	}

	hr = m_Device->CreateRenderTargetView(backBuffer, NULL, m_RenderTargetView.GetAddressOf());
	if (FAILED(hr))
	{
		BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create render target view."));
		return false;
	}

	backBuffer->Release();

	m_DeviceContext->OMSetRenderTargets(1, m_RenderTargetView.GetAddressOf(), NULL);

	D3D11_VIEWPORT viewport;
	ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));

	viewport.TopLeftX = 0;
	viewport.TopLeftY = 0;
	viewport.Width = static_cast<FLOAT>(width);
	viewport.Height = static_cast<FLOAT>(height);

	m_DeviceContext->RSSetViewports(1, &viewport);

	D3D11_RASTERIZER_DESC rasterizerDesc;
	ZeroMemory(&rasterizerDesc, sizeof(D3D11_RASTERIZER_DESC));

	rasterizerDesc.FillMode = D3D11_FILL_MODE::D3D11_FILL_SOLID;
	rasterizerDesc.CullMode = D3D11_CULL_MODE::D3D11_CULL_NONE;

	hr = m_Device->CreateRasterizerState(&rasterizerDesc, m_RasterizerState.GetAddressOf());
	if (FAILED(hr))
	{
		BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create rasterizer state."));
		return false;
	}

	D3D11_DEPTH_STENCIL_DESC depthStencilDesc;
	ZeroMemory(&depthStencilDesc, sizeof(D3D11_DEPTH_STENCIL_DESC));
	
	depthStencilDesc.DepthEnable = FALSE;
	depthStencilDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;

	hr = m_Device->CreateDepthStencilState(&depthStencilDesc, m_DepthStencilState.GetAddressOf());
	if (FAILED(hr))
	{
		BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create depth stencil state."));
		return false;
	}

	BB_INFO("DirectX initialized successful.");

	return true;
}
