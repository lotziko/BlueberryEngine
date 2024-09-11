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
		static void SetMatrix(const Matrix& matrix);
		static void DrawLine(const Vector3& start, const Vector3& end);
		static void DrawArc(const Vector3& center, const Vector3& normal, const Vector3& from, const float& angle, const float& radius);
		static void DrawBox(const Vector3& center, const Vector3& size);
		static void DrawCapsule(const Vector3& center, const float& height, const float& radius);
		static void DrawSphere(const Vector3& center, const float& radius);
		static void DrawDisc(const Vector3& center, const Vector3& normal, const float& radius);
		static void DrawFrustum(const Frustum& frustum);

	private:
		static void FlushLines();
		static void FlushArcs();

	private:
		struct Line
		{
			Vector3 start;
			Vector3 end;
			Color color;
		};

		struct Arc
		{
			Vector3 center;
			Vector3 normal;
			Vector3 from;
			float radius;
			float angle;
			Color color;
		};

	private:
		static inline Material* s_LineMaterial = nullptr;
		static inline Material* s_ArcMaterial = nullptr;
		static inline Color s_CurrentColor = Color();

		static inline GfxVertexBuffer* s_LineVertexBuffer = nullptr;
		static inline GfxVertexBuffer* s_ArcVertexBuffer = nullptr;

		static inline float* s_LineVertexData = nullptr;
		static inline float* s_LineVertexDataPtr = nullptr;
		static inline Line* s_Lines = nullptr;
		static inline float* s_ArcVertexData = nullptr;
		static inline float* s_ArcVertexDataPtr = nullptr;
		static inline Arc* s_Arcs = nullptr;
		static inline UINT s_LineCount = 0;
		static inline UINT s_ArcCount = 0;
	};
}