#pragma once

#include "Shader.h"
#include "Buffer.h"
#include "VertexLayout.h"
#include "Texture.h"
#include "ImGuiRenderer.h"
#include "Renderer2D.h"

class GraphicsDevice
{
public:
	virtual bool Initialize(int width, int height, void* data) = 0;

	virtual void ClearColor(const Color& color) const = 0;
	virtual void SwapBuffers() const = 0;

	virtual void SetViewport(int x, int y, int width, int height) = 0;
	virtual void ResizeBackbuffer(int width, int height) = 0;

	virtual bool CreateShader(const std::wstring& vertexShaderPath, const std::wstring& pixelShaderPath, Ref<Shader>& shader) = 0;
	virtual bool CreateVertexBuffer(const VertexLayout& layout, const UINT& vertexCount, Ref<VertexBuffer>& buffer) = 0;
	virtual bool CreateIndexBuffer(const UINT& indexCount, Ref<IndexBuffer>& buffer) = 0;
	virtual bool CreateConstantBuffer(const UINT& byteSize, Ref<ConstantBuffer>& buffer) = 0;
	virtual bool CreateTexture(const std::string& path, Ref<Texture>& texture) const = 0;
	virtual bool CreateImGuiRenderer(Ref<ImGuiRenderer>& renderer) const = 0;

	virtual void Draw(const int& vertices) const = 0;
	virtual void DrawIndexed(const int& indices) const = 0;

	virtual Matrix GetGPUMatrix(const Matrix& viewProjection) const = 0;

public:
	static Ref<GraphicsDevice> Create();
};