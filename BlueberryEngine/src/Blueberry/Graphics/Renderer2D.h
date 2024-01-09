#pragma once

#include "Blueberry\Math\Math.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Texture2D;
	class Material;
	class GfxVertexBuffer;
	class GfxIndexBuffer;
	class GfxConstantBuffer;

	class Renderer2D
	{
	public:
		Renderer2D() = default;
		~Renderer2D() = default;

		static bool Initialize();
		static void Shutdown();

		static void Begin(const Matrix& view, const Matrix& projection);
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
		static inline GfxVertexBuffer* s_VertexBuffer = nullptr;
		static inline GfxIndexBuffer* s_IndexBuffer = nullptr;
		static inline GfxConstantBuffer* s_ConstantBuffer = nullptr;

		static inline float* s_VertexData = nullptr;
		static inline float* s_VertexDataPtr = nullptr;
		static inline DrawingData* s_DrawingDatas = nullptr;
		static inline UINT s_QuadIndexCount = 0;
		static inline UINT s_DrawingDataCount = 0;
	};
}