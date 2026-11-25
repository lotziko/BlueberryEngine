#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Camera;
	class Material;
	class Mesh;
	class GfxBuffer;

	enum class GizmoEvent
	{
		Search,	// Find nearest handle
		Input,	// Process nearest handle
		Draw	// Draw
	};

	using HandleDistanceFunction = std::function<float(const Vector3&, const Quaternion&, float)>;
	using HandleDrawFunction = std::function<void(uint32_t, const Vector3&, const Quaternion&, float)>;

	class Gizmos
	{
	public:
		static bool Initialize();
		static void Shutdown();

		static void Begin(GizmoEvent event);
		static void End();
		static const Color& GetColor();
		static Color GetColor(float alpha);
		static void SetColor(const Color& color);
		static const Matrix& GetMatrix();
		static void SetMatrix(const Matrix& matrix);
		static bool GetIgnoreDepth();
		static void SetIgnoreDepth(bool ignoreDepth);
		static float GetThickness();
		static void SetThickness(float thickness);
		static void DrawLine(const Vector3& start, const Vector3& end);
		static void DrawArc(const Vector3& center, const Vector3& normal, const Vector3& from, float angle, float radius);
		static void DrawSolidArc(const Vector3& center, const Vector3& normal, const Vector3& from, float angle, float radius);
		static void DrawTriangle(const Vector3& a, const Vector3& b, const Vector3& c);
		static void DrawBox(const Vector3& center, const Vector3& size);
		static void DrawSolidBox(const Vector3& center, const Vector3& size);
		static void DrawCapsule(const Vector3& center, float height, float radius);
		static void DrawSphere(const Vector3& center, float radius);
		static void DrawSolidSphere(const Vector3& center, float radius);
		static void DrawDisc(const Vector3& center, const Vector3& normal, float radius);
		static void DrawFrustum(const Frustum& frustum);
		static void DrawMesh(const Vector3& position, const Quaternion& rotation, const Vector3& scale, Mesh* mesh);
		static void DrawMesh(const Vector3& position, const Quaternion& rotation, const Vector3& scale, GfxBuffer* vertexBuffer, GfxBuffer* indexBuffer);
		static void DrawSolidMesh(const Vector3& position, const Quaternion& rotation, const Vector3& scale, Mesh* mesh);
		static void DrawSolidMesh(const Vector3& position, const Quaternion& rotation, const Vector3& scale, GfxBuffer* vertexBuffer, GfxBuffer* indexBuffer);

	private:
		static void FlushLines();
		static void FlushArcs();
		static void FlushTriangles();
		static void BindMatrix(const Matrix& matrix);
		static void BindColorAndThickness(const Color& color, float thickness);

		// Handles
	public:
		static bool IsInUse();
		static void PushId(uint32_t id);
		static void PopId();
		static bool Slider1D(Vector3& position, const Vector3& direction);
		static bool Slider1D(const Vector3& position, const Vector3& direction, float& value);
		static bool Slider1D(Vector3& position, const Vector3& direction, float size, const HandleDistanceFunction& distanceFunction, const HandleDrawFunction& drawFunction);
		static bool Slider1D(uint32_t id, Vector3& position, const Vector3& direction, float size, const HandleDistanceFunction& distanceFunction, const HandleDrawFunction& drawFunction);
		static bool Slider2D(uint32_t id, Vector3& handlePosition, const Vector3& handleDirection, const Vector3& slideDirection1, const Vector3& slideDirection2, float size, const HandleDistanceFunction& distanceFunction, const HandleDrawFunction& drawFunction);
		static bool BoxBoundsHandle(Vector3& position, Vector3& size);
		static bool SphereBoundsHandle(Vector3& position, float& radius);
		static bool PositionHandle(Vector3& position, const Quaternion& rotation, bool isLocal);
		static bool RotationHandle(const Vector3& position, Quaternion& rotation, bool isLocal);
		static bool ScaleHandle(const Vector3& position, const Quaternion& rotation, Vector3& scale, bool isLocal);

		static float GetHandleSize(const Vector3& position);
		static void TrySetClosestHandle(uint32_t id, float distance);

	private:
		static uint32_t GetId(uint32_t id);
		static uint32_t GetUniqueId();

	private:
		struct Line
		{
			Vector3 positionA;
			Color colorA;
			Vector3 positionB;
			Color colorB;
		};

		struct Arc
		{
			Vector3 center;
			Vector3 normal;
			float radius;
			Vector3 from;
			float angle;
			Color color;
		};

		struct Triangle
		{
			Vector3 positionA;
			Color colorA;
			Vector3 positionB;
			Color colorB;
			Vector3 positionC;
			Color colorC;
		};

	private:
		static Material* s_GizmosMaterial;
		static GfxBuffer* s_LineVertexBuffer;
		static GfxBuffer* s_ArcVertexBuffer;
		static GfxBuffer* s_TriangleVertexBuffer;
		static GfxBuffer* s_ConstantBuffer;

		static const uint32_t MAX_LINES = 16384;
		static const uint32_t MAX_ARCS = 8192;
		static const uint32_t MAX_TRIANGLES = 8192;

		static Array<Line, MAX_LINES> s_Lines;
		static Array<Arc, MAX_ARCS> s_Arcs;
		static Array<Triangle, MAX_TRIANGLES> s_Triangles;

		static uint32_t s_LineCount;
		static uint32_t s_ArcCount;
		static uint32_t s_TriangleCount;
	};
}