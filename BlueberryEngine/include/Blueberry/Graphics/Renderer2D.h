#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Texture2D;
	class Material;
	class GfxBuffer;

	class Renderer2D
	{
	public:
		Renderer2D() = default;
		~Renderer2D() = default;

		static bool Initialize();
		static void Shutdown();

		static void Begin();
		static void End();
		static void Draw(const Matrix& transform, Texture2D* texture, Material* material, const Color& color, const int& sortingOrder);
		static void DrawImmediate(const Vector3& position, const Vector2& size, Texture2D* texture, Material* material, const Color& color);
		static void Flush();

	private:
		struct DrawingData
		{
			Matrix transform;
			Texture2D* texture;
			Material* material;
			Color color;
			int sortingOrder;
		};

		static bool SortBySortingOrder(DrawingData first, DrawingData second);

	private:
		static GfxBuffer* s_VertexBuffer;
		static GfxBuffer* s_IndexBuffer;

		static float* s_VertexData;
		static float* s_VertexDataPtr;
		static DrawingData* s_DrawingDatas;
		static uint32_t s_QuadIndexCount;
		static uint32_t s_DrawingDataCount;
	};
}