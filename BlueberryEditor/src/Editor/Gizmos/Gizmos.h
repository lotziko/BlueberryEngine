#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Material;
	class GfxBuffer;

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
		static void DrawArc(const Vector3& center, const Vector3& normal, const Vector3& from, float angle, float radius);
		static void DrawBox(const Vector3& center, const Vector3& size);
		static void DrawCapsule(const Vector3& center, float height, float radius);
		static void DrawSphere(const Vector3& center, float radius);
		static void DrawDisc(const Vector3& center, const Vector3& normal, float radius);
		static void DrawFrustum(const Frustum& frustum);
		static void DrawMesh(GfxBuffer* vertexBuffer, GfxBuffer* indexBuffer);

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
		static Material* s_GizmosMaterial;
		static Color s_CurrentColor;

		static GfxBuffer* s_LineVertexBuffer;
		static GfxBuffer* s_ArcVertexBuffer;

		static float* s_LineVertexData;
		static float* s_LineVertexDataPtr;
		static Line* s_Lines;
		static float* s_ArcVertexData;
		static float* s_ArcVertexDataPtr;
		static Arc* s_Arcs;
		static uint32_t s_LineCount;
		static uint32_t s_ArcCount;
	};
}