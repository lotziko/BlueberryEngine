#include "Renderer2D.h"

#include "..\Graphics\GfxBuffer.h"
#include "..\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\Mesh.h"

namespace Blueberry
{
	static Vector4 s_QuadVertexPositons[4];
	static Vector2 s_QuadTextureCoords[4];

	static const uint32_t MAX_QUADS = 4000; // TODO make polygons instead
	static const uint32_t MAX_SPRITES = 1024;
	static const uint32_t MAX_VERTICES = MAX_QUADS * 4;
	static const uint32_t MAX_INDICES = MAX_QUADS * 6;

	bool Renderer2D::Initialize()
	{
		VertexLayout layout = VertexLayout{}
			.Append(VertexAttribute::Position, 12)
			.Append(VertexAttribute::Color, 16)
			.Append(VertexAttribute::Texcoord0, 8)
			.Apply();

		int size = layout.GetSize();
		s_VertexData = BB_MALLOC_ARRAY(float, MAX_VERTICES * size / sizeof(float));

		if (!GfxDevice::CreateVertexBuffer(MAX_VERTICES, size, s_VertexBuffer))
		{
			return false;
		}

		uint32_t* indexData = BB_MALLOC_ARRAY(uint32_t, MAX_INDICES);
		uint32_t offset = 0;
		for (uint32_t i = 0; i < MAX_INDICES; i += 6)
		{
			indexData[i + 0] = offset + 0;
			indexData[i + 1] = offset + 1;
			indexData[i + 2] = offset + 2;

			indexData[i + 3] = offset + 2;
			indexData[i + 4] = offset + 3;
			indexData[i + 5] = offset + 0;

			offset += 4;
		}
		
		if (!GfxDevice::CreateIndexBuffer(MAX_INDICES, s_IndexBuffer))
		{
			return false;
		}
		s_IndexBuffer->SetData(indexData, MAX_INDICES);
		BB_FREE(indexData);

		s_DrawingDatas = BB_MALLOC_ARRAY(DrawingData, MAX_SPRITES);

		s_QuadVertexPositons[0] = { -0.5f, -0.5f, 0.0f, 1.0f };
		s_QuadVertexPositons[1] = { -0.5f, 0.5f, 0.0f, 1.0f };
		s_QuadVertexPositons[2] = { 0.5f, 0.5f, 0.0f, 1.0f };
		s_QuadVertexPositons[3] = { 0.5f, -0.5f, 0.0f, 1.0f };

		s_QuadTextureCoords[0] = { 0.0f, 0.0f };
		s_QuadTextureCoords[1] = { 0.0f, 1.0f };
		s_QuadTextureCoords[2] = { 1.0f, 1.0f };
		s_QuadTextureCoords[3] = { 1.0f, 0.0f };

		return true;
	}

	void Renderer2D::Shutdown()
	{
		BB_FREE(s_VertexData);
		BB_FREE(s_DrawingDatas);
	}

	void Renderer2D::Begin()
	{
		s_QuadIndexCount = 0;
		s_VertexDataPtr = s_VertexData;
	}

	void Renderer2D::End()
	{
		Flush();
	}

	void Renderer2D::Draw(const Matrix& transform, Texture2D* texture, Material* material, const Color& color, const int& sortingOrder)
	{
		if (s_DrawingDataCount >= MAX_SPRITES)
			Flush();

		if (texture == nullptr || !texture->IsDefaultState())
		{
			return;
		}

		if (material == nullptr || !material->IsDefaultState())
		{
			return;
		}

		s_DrawingDatas[s_DrawingDataCount] = { transform, texture, material, color, sortingOrder };
		++s_DrawingDataCount;
	}

	void Renderer2D::DrawImmediate(const Vector3& position, const Vector2& size, Texture2D* texture, Material* material, const Color& color)
	{
		if (s_QuadIndexCount > 0)
			Flush();
		Draw(Matrix::CreateTranslation(position) * Matrix::CreateScale(size.x, size.y, 1), texture, material, color, 0);
		Flush();
	}

	void Renderer2D::Flush()
	{
		static size_t baseMapId = TO_HASH("_BaseMap");

		if (s_DrawingDataCount == 0)
			return;

		std::sort(s_DrawingDatas, s_DrawingDatas + s_DrawingDataCount, SortBySortingOrder);

		// TODO non rectangle sprites
		// Fill vertices
		for (uint32_t i = 0; i < s_DrawingDataCount; i++)
		{
			DrawingData data = s_DrawingDatas[i];
			Matrix transform = data.transform;
			Texture2D* texture = data.texture;
			Color color = data.color;

			for (int j = 0; j < 4; j++)
			{
				auto position = Vector4::Transform(s_QuadVertexPositons[j] * Vector4(static_cast<float>(texture->GetWidth()) / 32, static_cast<float>(texture->GetHeight()) / 32, 1, 1), transform);
				
				s_VertexDataPtr[0] = position.x;
				s_VertexDataPtr[1] = position.y;
				s_VertexDataPtr[2] = position.z;

				s_VertexDataPtr[3] = color.x;
				s_VertexDataPtr[4] = color.y;
				s_VertexDataPtr[5] = color.z;
				s_VertexDataPtr[6] = color.w;

				s_VertexDataPtr[7] = s_QuadTextureCoords[j].x;
				s_VertexDataPtr[8] = s_QuadTextureCoords[j].y;

				s_VertexDataPtr += 9;
			}
			s_QuadIndexCount += 6;
		}

		s_VertexBuffer->SetData(s_VertexData, s_QuadIndexCount / 6 * 4);

		// Draw quads
		Material* currentMaterial = s_DrawingDatas->material;
		Texture2D* currentTexture = s_DrawingDatas->texture;
		uint32_t indexOffset = 0;
		uint32_t indexCount = 0;
		for (uint32_t i = 0; i < s_DrawingDataCount; i++)
		{
			DrawingData data = s_DrawingDatas[i];
			Material* material = data.material;
			Texture2D* texture = data.texture;
			if (material != currentMaterial || texture != currentTexture)
			{
				GfxDevice::SetGlobalTexture(baseMapId, currentTexture->Get());
				GfxDevice::Draw(GfxDrawingOperation(s_VertexBuffer, s_IndexBuffer, currentMaterial, 0, indexCount, indexOffset, 0, Topology::TriangleList));

				currentMaterial = material;
				currentTexture = texture;
				indexOffset += indexCount;
				indexCount = 0;
			}
			indexCount += 6;
		}

		if (indexCount > 0)
		{
			GfxDevice::SetGlobalTexture(baseMapId, currentTexture->Get());
			GfxDevice::Draw(GfxDrawingOperation(s_VertexBuffer, s_IndexBuffer, currentMaterial, 0, indexCount, indexOffset, 0, Topology::TriangleList));
		}

		s_QuadIndexCount = 0;
		s_VertexDataPtr = s_VertexData;
		s_DrawingDataCount = 0;
	}

	bool Renderer2D::SortBySortingOrder(DrawingData first, DrawingData second)
	{
		return first.sortingOrder < second.sortingOrder;
	}
}