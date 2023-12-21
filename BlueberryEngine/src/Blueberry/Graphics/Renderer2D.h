#pragma once

#include "Blueberry\Math\Math.h"
#include "Blueberry\Core\WeakObjectPtr.h"

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
		Renderer2D();
		~Renderer2D() = default;

		bool Initialize();
		void Shutdown();

		void Begin(const Matrix& view, const Matrix& projection);
		void End();
		void Draw(const Matrix& transform, Texture2D* texture, Material* material, const Color& color);
		void DrawImmediate(const Vector3& position, const Vector2& size, Texture2D* texture, Material* material, const Color& color);
		void Flush();

	private:
		WeakObjectPtr<Material> m_Material;
		GfxVertexBuffer* m_VertexBuffer;
		GfxIndexBuffer* m_IndexBuffer;
		GfxConstantBuffer* m_ConstantBuffer;

		float* m_VertexData = nullptr;
		float* m_VertexDataPtr = nullptr;
		UINT m_QuadIndexCount = 0;

		Vector4 m_QuadVertexPositons[4];
		Vector2 m_QuadTextureCoords[4];

		static const UINT MAX_QUADS = 4000;
		static const UINT MAX_VERTICES = MAX_QUADS * 4;
		static const UINT MAX_INDICES = MAX_QUADS * 6;
	};
}