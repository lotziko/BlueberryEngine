#pragma once

#include "Blueberry\Math\Math.h"

namespace Blueberry
{
	class GraphicsDevice;
	class Texture;
	class VertexBuffer;
	class IndexBuffer;
	class ConstantBuffer;
	class Shader;

	class Renderer2D
	{
	public:
		Renderer2D();
		~Renderer2D() = default;

		bool Initialize();
		void Shutdown();

		void Begin(const Matrix& view, const Matrix& projection);
		void End();
		void Draw(const Matrix& transform, Texture* texture, const Color& color);
		void DrawImmediate(const Vector3& position, const Vector2& size, Texture* texture, const Color& color);
		void Flush();

	private:
		Ref<VertexBuffer> m_VertexBuffer;
		Ref<IndexBuffer> m_IndexBuffer;
		Ref<ConstantBuffer> m_ConstantBuffer;
		Ref<Shader> m_DefaultShader;

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