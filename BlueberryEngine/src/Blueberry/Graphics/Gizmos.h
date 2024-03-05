#pragma once

namespace Blueberry
{
	class Material;
	class GfxVertexBuffer;
	class GfxIndexBuffer;
	class GfxConstantBuffer;

	class Gizmos
	{
	public:
		static bool Initialize();
		static void Shutdown();

		static void Begin();
		static void End();
		static void SetColor(const Color& color);
		static void DrawLine(const Vector3& start, const Vector3& end);
		static void Flush();

	private:
		struct Line
		{
			Vector3 start;
			Vector3 end;
			Color color;
		};

	private:
		static inline Material* s_LineMaterial = nullptr;
		static inline Color s_CurrentColor = Color();

		static inline GfxVertexBuffer* s_VertexBuffer = nullptr;
		static inline GfxIndexBuffer* s_IndexBuffer = nullptr;

		static inline float* s_VertexData = nullptr;
		static inline float* s_VertexDataPtr = nullptr;
		static inline Line* s_Lines = nullptr;
		static inline UINT s_LineCount = 0;
	};
}