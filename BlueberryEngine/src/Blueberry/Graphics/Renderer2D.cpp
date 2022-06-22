#include "bbpch.h"
#include "Renderer2D.h"

#include "Blueberry\Core\GlobalServices.h"
#include "Blueberry\Graphics\GraphicsDevice.h"
#include "Blueberry\Graphics\Shader.h"

namespace Blueberry
{
	struct CONSTANTS
	{
		Matrix viewProjectionMatrix;
	};

	Renderer2D::Renderer2D()
	{
	}

	bool Renderer2D::Initialize()
	{
		VertexLayout layout = VertexLayout{}
			.Append(VertexLayout::Position3D)
			.Append(VertexLayout::Float4Color)
			.Append(VertexLayout::TextureCoord);

		int size = layout.GetSize();
		m_VertexData = new float[MAX_VERTICES * size / sizeof(float)];

		if (!g_GraphicsDevice->CreateVertexBuffer(layout, MAX_VERTICES, m_VertexBuffer))
		{
			return false;
		}

		UINT* indexData = new UINT[MAX_INDICES];
		UINT offset = 0;
		for (UINT i = 0; i < MAX_INDICES; i += 6)
		{
			indexData[i + 0] = offset + 0;
			indexData[i + 1] = offset + 1;
			indexData[i + 2] = offset + 2;

			indexData[i + 3] = offset + 2;
			indexData[i + 4] = offset + 3;
			indexData[i + 5] = offset + 0;

			offset += 4;
		}

		if (!g_GraphicsDevice->CreateIndexBuffer(MAX_INDICES, m_IndexBuffer))
		{
			return false;
		}
		m_IndexBuffer->SetData(indexData, MAX_INDICES);
		delete[] indexData;

		if (!g_GraphicsDevice->CreateConstantBuffer(sizeof(CONSTANTS) * 1, m_ConstantBuffer))
		{
			return false;
		}

		if (!g_AssetManager->Load<Shader>("assets/standard", m_DefaultShader))
		{
			return false;
		}

		m_QuadVertexPositons[0] = { -0.5f, -0.5f, 0.0f, 1.0f };
		m_QuadVertexPositons[1] = { -0.5f, 0.5f, 0.0f, 1.0f };
		m_QuadVertexPositons[2] = { 0.5f, 0.5f, 0.0f, 1.0f };
		m_QuadVertexPositons[3] = { 0.5f, -0.5f, 0.0f, 1.0f };

		m_QuadTextureCoords[0] = { 0.0f, 0.0f };
		m_QuadTextureCoords[1] = { 0.0f, 1.0f };
		m_QuadTextureCoords[2] = { 1.0f, 1.0f };
		m_QuadTextureCoords[3] = { 1.0f, 0.0f };

		return true;
	}

	void Renderer2D::Shutdown()
	{
		delete[] m_VertexData;
	}

	void Renderer2D::Begin(const Matrix& view, const Matrix& projection)
	{
		CONSTANTS constants[] =
		{
			{ g_GraphicsDevice->GetGPUMatrix(view * projection) }
		};

		m_ConstantBuffer->SetData(reinterpret_cast<char*>(&constants), sizeof(constants));

		m_QuadIndexCount = 0;
		m_VertexDataPtr = m_VertexData;
	}

	void Renderer2D::End()
	{
		Flush();
	}

	void Renderer2D::Draw(const Matrix& transform, Texture* texture, const Color& color)
	{
		if (m_QuadIndexCount >= MAX_INDICES)
			Flush();

		if (texture == nullptr)
		{
			return;
		}
		texture->Bind();

		for (int i = 0; i < 4; i++)
		{
			auto position = Vector4::Transform(m_QuadVertexPositons[i], transform);

			m_VertexDataPtr[0] = position.x;
			m_VertexDataPtr[1] = position.y;
			m_VertexDataPtr[2] = position.z;

			m_VertexDataPtr[3] = color.x;
			m_VertexDataPtr[4] = color.y;
			m_VertexDataPtr[5] = color.z;
			m_VertexDataPtr[6] = color.w;

			m_VertexDataPtr[7] = m_QuadTextureCoords[i].x;
			m_VertexDataPtr[8] = m_QuadTextureCoords[i].y;

			m_VertexDataPtr += 9;
		}

		m_QuadIndexCount += 6;
	}

	void Renderer2D::DrawImmediate(const Vector3& position, const Vector2& size, Texture* texture, const Color& color)
	{
		if (m_QuadIndexCount > 0)
			Flush();
		Draw(Matrix::CreateTranslation(position) * Matrix::CreateScale(size.x, size.y, 1), texture, color);
		Flush();
	}

	void Renderer2D::Flush()
	{
		if (m_QuadIndexCount == 0)
			return;

		m_VertexBuffer->SetData(m_VertexData, m_QuadIndexCount / 6 * 4);

		m_VertexBuffer->Bind();
		m_IndexBuffer->Bind();
		m_ConstantBuffer->Bind();
		m_DefaultShader->Bind();
		g_GraphicsDevice->DrawIndexed(m_QuadIndexCount);

		m_QuadIndexCount = 0;
		m_VertexDataPtr = m_VertexData;
	}
}