#include "bbpch.h"
#include "Renderer2D.h"

#include "Blueberry\Core\GlobalServices.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\Mesh.h"

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

		m_DrawingDatas = new DrawingData[MAX_SPRITES];

		if (!g_GraphicsDevice->CreateConstantBuffer(sizeof(CONSTANTS) * 1, m_ConstantBuffer))
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
		delete[] m_DrawingDatas;
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

	void Renderer2D::Draw(const Matrix& transform, Texture2D* texture, Material* material, const Color& color, const int& sortingOrder)
	{
		if (m_DrawingDataCount >= MAX_SPRITES)
			Flush();

		if (texture == nullptr)
		{
			return;
		}

		if (material == nullptr)
		{
			return;
		}

		m_DrawingDatas[m_DrawingDataCount] = { transform, texture, material, color, sortingOrder };
		++m_DrawingDataCount;
	}

	void Renderer2D::DrawImmediate(const Vector3& position, const Vector2& size, Texture2D* texture, Material* material, const Color& color)
	{
		if (m_QuadIndexCount > 0)
			Flush();
		Draw(Matrix::CreateTranslation(position) * Matrix::CreateScale(size.x, size.y, 1), texture, material, color, 0);
		Flush();
	}

	void Renderer2D::Flush()
	{
		static size_t baseMapId = TO_HASH("_BaseMap");

		if (m_DrawingDataCount == 0)
			return;

		std::sort(m_DrawingDatas, m_DrawingDatas + m_DrawingDataCount, SortBySortingOrder);

		g_GraphicsDevice->SetGlobalConstantBuffer(std::hash<std::string>()("PerDrawData"), m_ConstantBuffer);

		// TODO non rectangle sprites
		// Fill vertices
		for (int i = 0; i < m_DrawingDataCount; i++)
		{
			DrawingData data = m_DrawingDatas[i];
			Matrix transform = data.transform;
			Texture2D* texture = data.texture;
			Color color = data.color;

			for (int j = 0; j < 4; j++)
			{
				auto position = Vector4::Transform(m_QuadVertexPositons[j] * Vector4(texture->GetWidth() / 32, texture->GetHeight() / 32, 1, 1), transform);
				
				m_VertexDataPtr[0] = position.x;
				m_VertexDataPtr[1] = position.y;
				m_VertexDataPtr[2] = position.z;

				m_VertexDataPtr[3] = color.x;
				m_VertexDataPtr[4] = color.y;
				m_VertexDataPtr[5] = color.z;
				m_VertexDataPtr[6] = color.w;

				m_VertexDataPtr[7] = m_QuadTextureCoords[j].x;
				m_VertexDataPtr[8] = m_QuadTextureCoords[j].y;

				m_VertexDataPtr += 9;
			}
			m_QuadIndexCount += 6;
		}

		m_VertexBuffer->SetData(m_VertexData, m_QuadIndexCount / 6 * 4);

		// Draw quads
		Material* currentMaterial = m_DrawingDatas->material;
		Texture2D* currentTexture = m_DrawingDatas->texture;
		UINT indexOffset = 0;
		UINT indexCount = 0;
		for (int i = 0; i < m_DrawingDataCount; i++)
		{
			DrawingData data = m_DrawingDatas[i];
			Material* material = data.material;
			Texture2D* texture = data.texture;
			if (material != currentMaterial || texture != currentTexture)
			{
				g_GraphicsDevice->SetGlobalTexture(baseMapId, currentTexture->m_Texture);
				g_GraphicsDevice->Draw(GfxDrawingOperation(m_VertexBuffer, m_IndexBuffer, currentMaterial, indexCount, indexOffset, Topology::TriangleList));

				currentMaterial = material;
				currentTexture = texture;
				indexOffset += indexCount;
				indexCount = 0;
			}
			indexCount += 6;
		}

		if (indexCount > 0)
		{
			g_GraphicsDevice->SetGlobalTexture(baseMapId, currentTexture->m_Texture);
			g_GraphicsDevice->Draw(GfxDrawingOperation(m_VertexBuffer, m_IndexBuffer, currentMaterial, indexCount, indexOffset, Topology::TriangleList));
		}

		m_QuadIndexCount = 0;
		m_VertexDataPtr = m_VertexData;
		m_DrawingDataCount = 0;
	}

	bool Renderer2D::SortBySortingOrder(DrawingData first, DrawingData second)
	{
		return first.sortingOrder < second.sortingOrder;
	}
}