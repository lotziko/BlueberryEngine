#include "Gizmos.h"

#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\VertexLayout.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\Buffers\PerDrawDataConstantBuffer.h"
#include "Blueberry\Tools\CRCHelper.h"

#include "Editor\Misc\SceneHelper.h"
#include "GizmoDataConstantBuffer.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	Material* Gizmos::s_GizmosMaterial = nullptr;
	GfxBuffer* Gizmos::s_LineVertexBuffer = nullptr;
	GfxBuffer* Gizmos::s_ArcVertexBuffer = nullptr;
	GfxBuffer* Gizmos::s_TriangleVertexBuffer = nullptr;

	Array<Gizmos::Line, Gizmos::MAX_LINES> Gizmos::s_Lines = {};
	Array<Gizmos::Arc, Gizmos::MAX_ARCS> Gizmos::s_Arcs = {};
	Array<Gizmos::Triangle, Gizmos::MAX_TRIANGLES> Gizmos::s_Triangles = {};

	uint32_t Gizmos::s_LineCount = 0;
	uint32_t Gizmos::s_ArcCount = 0;
	uint32_t Gizmos::s_TriangleCount = 0;

	static VertexLayout s_LineLayout;
	static VertexLayout s_ArcLayout;
	static VertexLayout s_TriangleLayout;
	static VertexLayout s_MeshLayout;

	struct Context
	{
		GizmoEvent event;
		// Gizmos
		Color color;
		Color bindedColor = Color(0, 0, 0, 0);
		Matrix matrix;
		Matrix inverseMatrix;
		Matrix bindedMatrix = Matrix(Vector4(0, 0, 0, 0), Vector4(0, 0, 0, 0), Vector4(0, 0, 0, 0), Vector4(0, 0, 0, 0));
		bool ignoreDepth;
		float thickness;
		float bindedThickness = 0;
		// Handles
		Camera* camera = nullptr;
		Vector2 mousePosition;
		bool isMouseDown;
		bool isMousePressed;
		bool isMouseDragging;
		float closestDistance = FLT_MAX;
		uint32_t closestHandle = UINT32_MAX;
		uint32_t activeHandle = UINT32_MAX;
		List<uint32_t> idStack;
		uint32_t maxUniqueId = 0;
	};

	static Context s_Context;

	constexpr uint32_t GetStrId(const char* str);

	static uint32_t s_AxisHandleId[7] = { GetStrId("XSlider"), GetStrId("YSlider"), GetStrId("ZSlider"), GetStrId("YZSlider"), GetStrId("XZSlider"), GetStrId("XYSlider"), GetStrId("CameraSlider") };
	static Color s_AxisHandleColor[7] = { Color(1, 0, 0, 1), Color(0, 1, 0, 1), Color(0, 0, 1, 1), Color(1, 0, 0, 1), Color(0, 1, 0, 1), Color(0, 0, 1, 1), Color(0.5f, 0.5f, 0.5f, 1) };
	static Vector3 s_AxisHandleDirection[4] = { Vector3::Right, Vector3::Up, Vector3::Forward, Vector3::Zero };
	static Vector3 s_PlaneHandleDirection[3] = { Vector3::Right, Vector3::Up, Vector3::Forward };
	static Vector3 s_PlaneSlideDirection1[3] = { Vector3::Up, Vector3::Right, Vector3::Right };
	static Vector3 s_PlaneSlideDirection2[3] = { Vector3::Forward, Vector3::Forward, Vector3::Up };
	static Vector3 s_PlaneOffset[3] = { Vector3(0, 1, 1), Vector3(1, 0, 1), Vector3(1, 1, 0) };
	static float s_TransformGizmoScale = 0.95f;
	static Vector3 s_StartScale;

	bool Gizmos::Initialize()
	{
		Shader* gizmosShader = static_cast<Shader*>(AssetLoader::Load("assets/shaders/Gizmos.shader"));
		if (gizmosShader == nullptr)
		{
			BB_ERROR("Failed to load gizmos shader.");
			return false;
		}
		s_GizmosMaterial = Material::Create(gizmosShader);

		s_LineLayout = VertexLayout{}
			.Append(VertexAttribute::Position, 12)
			.Append(VertexAttribute::Color, 16)
			.Apply();

		BufferProperties lineBufferProperties = {};
		lineBufferProperties.elementCount = MAX_LINES * 2;
		lineBufferProperties.elementSize = s_LineLayout.GetSize();
		lineBufferProperties.usageFlags = BufferUsageFlags::VertexBuffer | BufferUsageFlags::CPUWritable;

		if (!GfxDevice::CreateBuffer(lineBufferProperties, s_LineVertexBuffer))
		{
			return false;
		}

		s_ArcLayout = VertexLayout{}
			.Append(VertexAttribute::Position, 12)
			.Append(VertexAttribute::Normal, 16)
			.Append(VertexAttribute::Tangent, 16)
			.Append(VertexAttribute::Color, 16)
			.Apply();

		BufferProperties arcBufferProperties = {};
		arcBufferProperties.elementCount = MAX_ARCS;
		arcBufferProperties.elementSize = s_ArcLayout.GetSize();
		arcBufferProperties.usageFlags = BufferUsageFlags::VertexBuffer | BufferUsageFlags::CPUWritable;

		if (!GfxDevice::CreateBuffer(arcBufferProperties, s_ArcVertexBuffer))
		{
			return false;
		}

		s_TriangleLayout = VertexLayout{}
			.Append(VertexAttribute::Position, 12)
			.Append(VertexAttribute::Color, 16)
			.Apply();

		BufferProperties triangleBufferProperties = {};
		triangleBufferProperties.elementCount = MAX_TRIANGLES * 3;
		triangleBufferProperties.elementSize = s_TriangleLayout.GetSize();
		triangleBufferProperties.usageFlags = BufferUsageFlags::VertexBuffer | BufferUsageFlags::CPUWritable;

		if (!GfxDevice::CreateBuffer(triangleBufferProperties, s_TriangleVertexBuffer))
		{
			return false;
		}

		s_MeshLayout = VertexLayout{}
			.Append(VertexAttribute::Position, 12)
			.Apply();

		return true;
	}

	void Gizmos::Shutdown()
	{
		Material::Destroy(s_GizmosMaterial);
		delete s_TriangleVertexBuffer;
	}

	void Gizmos::Begin(GizmoEvent event)
	{
		s_Context.event = event;
		s_Context.matrix = Matrix::Identity;
		s_Context.inverseMatrix = Matrix::Identity;
		s_Context.maxUniqueId = 0;
		switch (event)
		{
		case GizmoEvent::Search:
		{
			ImVec2 mousePos = ImGui::GetMousePos();
			ImVec2 pos = ImGui::GetCursorScreenPos();
			s_Context.camera = Camera::GetCurrent();
			s_Context.mousePosition = Vector2(mousePos.x - pos.x, mousePos.y - pos.y);
			s_Context.isMouseDown = ImGui::IsMouseDown(0);
			s_Context.isMousePressed = ImGui::IsMouseClicked(0);
			s_Context.isMouseDragging = ImGui::IsMouseDragging(0);
			s_Context.closestDistance = FLT_MAX;
			s_Context.closestHandle = UINT32_MAX;
			if (ImGui::IsMouseReleased(0))
			{
				s_Context.activeHandle = UINT32_MAX;
			}
			s_Context.idStack.clear();
		}
		break;
		case GizmoEvent::Draw:
		{
			s_Context.color = Color(1.0f, 1.0f, 1.0f, 1.0f);
			s_Context.ignoreDepth = false;
			s_Context.bindedMatrix.m[0][0] = FLT_MAX;
		}
		break;
		}
	}

	void Gizmos::End()
	{
		if (s_Context.event == GizmoEvent::Draw)
		{
			FlushLines();
			FlushArcs();
			FlushTriangles();
		}
	}

	const Color& Gizmos::GetColor()
	{
		return s_Context.color;
	}

	Color Gizmos::GetColor(float alpha)
	{
		Color color = s_Context.color;
		color.w *= alpha;
		return color;
	}

	void Gizmos::SetColor(const Color& color)
	{
		s_Context.color = color;
	}

	const Matrix& Gizmos::GetMatrix()
	{
		return s_Context.matrix;
	}

	void Gizmos::SetMatrix(const Matrix& matrix)
	{
		if (s_Context.matrix == matrix)
		{
			return;
		}
		if (s_Context.event == GizmoEvent::Draw)
		{
			if (!s_Lines.empty())
			{
				FlushLines();
			}
			if (!s_Arcs.empty())
			{
				FlushArcs();
			}
			if (!s_Triangles.empty())
			{
				FlushTriangles();
			}
		}
		s_Context.matrix = matrix;
		s_Context.inverseMatrix = matrix.Invert();
	}

	bool Gizmos::GetIgnoreDepth()
	{
		return s_Context.ignoreDepth;
	}

	void Gizmos::SetIgnoreDepth(bool ignoreDepth)
	{
		if (s_Context.ignoreDepth == ignoreDepth)
		{
			return;
		}
		if (s_Context.event == GizmoEvent::Draw)
		{
			if (!s_Lines.empty())
			{
				FlushLines();
			}
			if (!s_Arcs.empty())
			{
				FlushArcs();
			}
			if (!s_Triangles.empty())
			{
				FlushTriangles();
			}
		}
		s_Context.ignoreDepth = ignoreDepth;
	}

	float Gizmos::GetThickness()
	{
		return s_Context.thickness;
	}

	void Gizmos::SetThickness(float thickness)
	{
		if (s_Context.thickness == thickness)
		{
			return;
		}
		if (s_Context.event == GizmoEvent::Draw)
		{
			if (!s_Lines.empty())
			{
				FlushLines();
			}
		}
		s_Context.thickness = thickness;
	}

	void Gizmos::DrawLine(const Vector3& start, const Vector3& end)
	{
		if (s_Context.event != GizmoEvent::Draw)
		{
			return;
		}
		if (s_LineCount + 1 >= MAX_LINES)
		{
			FlushLines();
		}
		s_Lines[s_LineCount++] = { start, s_Context.color, end, s_Context.color };
	}

	void Gizmos::DrawArc(const Vector3& center, const Vector3& normal, const Vector3& from, float angle, float radius)
	{
		if (s_Context.event != GizmoEvent::Draw)
		{
			return;
		}
		if (s_ArcCount + 1 >= MAX_ARCS)
		{
			FlushArcs();
		}
		s_Arcs[s_ArcCount++] = { center, normal, radius, from, angle, s_Context.color };
	}

	void Gizmos::DrawSolidArc(const Vector3& center, const Vector3& normal, const Vector3& from, float angle, float radius)
	{
		if (s_Context.event != GizmoEvent::Draw)
		{
			return;
		}
		Vector3 bitangent = normal.Cross(from);
		Vector3 points[37];
		float segments = 36.0;
		for (int i = 0; i <= 36; ++i)
		{
			float u = cos(Math::ToRadians(i / segments * angle)) * radius;
			float v = sin(Math::ToRadians(i / segments * angle)) * radius;
			points[i] = center + from * u + bitangent * v;
		}
		for (int i = 1; i < 37; ++i)
		{
			DrawTriangle(center, points[i - 1], points[i]);
			DrawTriangle(center, points[i], points[i - 1]);
		}
	}

	void Gizmos::DrawTriangle(const Vector3& a, const Vector3& b, const Vector3& c)
	{
		if (s_Context.event != GizmoEvent::Draw)
		{
			return;
		}
		if (s_TriangleCount + 1 >= MAX_TRIANGLES)
		{
			FlushTriangles();
		}
		s_Triangles[s_TriangleCount++] = { a, s_Context.color, b, s_Context.color, c, s_Context.color };
	}

	void Gizmos::DrawBox(const Vector3& center, const Vector3& size)
	{
		if (s_Context.event != GizmoEvent::Draw)
		{
			return;
		}
		if (s_LineCount + 12 >= MAX_LINES)
		{
			FlushLines();
		}
		float halfX = size.x * 0.5f;
		float halfY = size.y * 0.5f;
		float halfZ = size.z * 0.5f;

		Vector3 c0 = Vector3(-halfX, -halfY, -halfZ);
		Vector3 c1 = Vector3(halfX, -halfY, -halfZ);
		Vector3 c2 = Vector3(halfX, -halfY, halfZ);
		Vector3 c3 = Vector3(-halfX, -halfY, halfZ);

		Vector3 c4 = Vector3(-halfX, halfY, -halfZ);
		Vector3 c5 = Vector3(halfX, halfY, -halfZ);
		Vector3 c6 = Vector3(halfX, halfY, halfZ);
		Vector3 c7 = Vector3(-halfX, halfY, halfZ);

		s_Lines[s_LineCount++] = { center + c0, s_Context.color, center + c1, s_Context.color };
		s_Lines[s_LineCount++] = { center + c1, s_Context.color, center + c2, s_Context.color };
		s_Lines[s_LineCount++] = { center + c2, s_Context.color, center + c3, s_Context.color };
		s_Lines[s_LineCount++] = { center + c3, s_Context.color, center + c0, s_Context.color };

		s_Lines[s_LineCount++] = { center + c4, s_Context.color, center + c5, s_Context.color };
		s_Lines[s_LineCount++] = { center + c5, s_Context.color, center + c6, s_Context.color };
		s_Lines[s_LineCount++] = { center + c6, s_Context.color, center + c7, s_Context.color };
		s_Lines[s_LineCount++] = { center + c7, s_Context.color, center + c4, s_Context.color };

		s_Lines[s_LineCount++] = { center + c0, s_Context.color, center + c4, s_Context.color };
		s_Lines[s_LineCount++] = { center + c1, s_Context.color, center + c5, s_Context.color };
		s_Lines[s_LineCount++] = { center + c2, s_Context.color, center + c6, s_Context.color };
		s_Lines[s_LineCount++] = { center + c3, s_Context.color, center + c7, s_Context.color };
	}

	void Gizmos::DrawSolidBox(const Vector3& center, const Vector3& size)
	{
		if (s_Context.event != GizmoEvent::Draw)
		{
			return;
		}
		if (s_TriangleCount + 12 >= MAX_TRIANGLES)
		{
			FlushTriangles();
		}
		float halfX = size.x * 0.5f;
		float halfY = size.y * 0.5f;
		float halfZ = size.z * 0.5f;

		Vector3 c0 = Vector3(-halfX, -halfY, -halfZ);
		Vector3 c1 = Vector3(halfX, -halfY, -halfZ);
		Vector3 c2 = Vector3(halfX, -halfY, halfZ);
		Vector3 c3 = Vector3(-halfX, -halfY, halfZ);

		Vector3 c4 = Vector3(-halfX, halfY, -halfZ);
		Vector3 c5 = Vector3(halfX, halfY, -halfZ);
		Vector3 c6 = Vector3(halfX, halfY, halfZ);
		Vector3 c7 = Vector3(-halfX, halfY, halfZ);

		s_Triangles[s_TriangleCount++] = { center + c0, s_Context.color, center + c1, s_Context.color, center + c2, s_Context.color };
		s_Triangles[s_TriangleCount++] = { center + c2, s_Context.color, center + c3, s_Context.color, center + c0, s_Context.color };
		s_Triangles[s_TriangleCount++] = { center + c4, s_Context.color, center + c5, s_Context.color, center + c6, s_Context.color };
		s_Triangles[s_TriangleCount++] = { center + c6, s_Context.color, center + c7, s_Context.color, center + c4, s_Context.color };
		s_Triangles[s_TriangleCount++] = { center + c0, s_Context.color, center + c1, s_Context.color, center + c5, s_Context.color };
		s_Triangles[s_TriangleCount++] = { center + c5, s_Context.color, center + c4, s_Context.color, center + c0, s_Context.color };
		s_Triangles[s_TriangleCount++] = { center + c3, s_Context.color, center + c2, s_Context.color, center + c6, s_Context.color };
		s_Triangles[s_TriangleCount++] = { center + c6, s_Context.color, center + c7, s_Context.color, center + c3, s_Context.color };
		s_Triangles[s_TriangleCount++] = { center + c0, s_Context.color, center + c4, s_Context.color, center + c7, s_Context.color };
		s_Triangles[s_TriangleCount++] = { center + c7, s_Context.color, center + c3, s_Context.color, center + c0, s_Context.color };
		s_Triangles[s_TriangleCount++] = { center + c1, s_Context.color, center + c5, s_Context.color, center + c6, s_Context.color };
		s_Triangles[s_TriangleCount++] = { center + c6, s_Context.color, center + c2, s_Context.color, center + c1, s_Context.color };
	}

	void Gizmos::DrawCapsule(const Vector3& center, float height, float radius)
	{
		if (s_Context.event != GizmoEvent::Draw)
		{
			return;
		}
		if (s_LineCount + 4 >= MAX_LINES)
		{
			FlushArcs();
		}
		if (s_ArcCount + 6 >= MAX_ARCS)
		{
			FlushArcs();
		}
		float halfHeight = height / 2;
		float topHeight = center.y + halfHeight;
		float bottomHeight = center.y - halfHeight;

		s_Lines[s_LineCount++] = { Vector3(center.x - radius, topHeight, center.z), s_Context.color, Vector3(center.x - radius, bottomHeight, center.z), s_Context.color };
		s_Lines[s_LineCount++] = { Vector3(center.x + radius, topHeight, center.z), s_Context.color, Vector3(center.x + radius, bottomHeight, center.z), s_Context.color };
		s_Lines[s_LineCount++] = { Vector3(center.x, topHeight, center.z - radius), s_Context.color, Vector3(center.x, bottomHeight, center.z - radius), s_Context.color };
		s_Lines[s_LineCount++] = { Vector3(center.x, topHeight, center.z + radius), s_Context.color, Vector3(center.x, bottomHeight, center.z + radius), s_Context.color };

		s_Arcs[s_ArcCount++] = { Vector3(center.x, topHeight, center.z), -Vector3::UnitX, radius, Vector3::UnitZ, 180, s_Context.color };
		s_Arcs[s_ArcCount++] = { Vector3(center.x, topHeight, center.z), Vector3::UnitZ, radius, Vector3::UnitX, 180, s_Context.color };
		s_Arcs[s_ArcCount++] = { Vector3(center.x, bottomHeight, center.z), Vector3::UnitX, radius, Vector3::UnitZ, 180, s_Context.color };
		s_Arcs[s_ArcCount++] = { Vector3(center.x, bottomHeight, center.z), -Vector3::UnitZ, radius, Vector3::UnitX, 180, s_Context.color };

		s_Arcs[s_ArcCount++] = { Vector3(center.x, topHeight, center.z), Vector3::UnitY, radius, Vector3::UnitX, 360, s_Context.color };
		s_Arcs[s_ArcCount++] = { Vector3(center.x, bottomHeight, center.z), Vector3::UnitY, radius, Vector3::UnitX, 360, s_Context.color };
	}

	void Gizmos::DrawSphere(const Vector3& center, float radius)
	{
		if (s_Context.event != GizmoEvent::Draw)
		{
			return;
		}
		if (s_ArcCount + 3 >= MAX_ARCS)
		{
			FlushArcs();
		}
		s_Arcs[s_ArcCount++] = { center, Vector3(0, 1, 0), radius, Vector3::UnitX, 360, s_Context.color };
		s_Arcs[s_ArcCount++] = { center, Vector3(0, 0, 1), radius, Vector3::UnitX, 360, s_Context.color };
		s_Arcs[s_ArcCount++] = { center, Vector3(1, 0, 0), radius, Vector3::UnitZ, 360, s_Context.color };
	}

	void Gizmos::DrawSolidSphere(const Vector3& center, float radius)
	{
		DrawSolidMesh(center, Quaternion::Identity, Vector3(radius, radius, radius) * 2.0f, StandardMeshes::GetSphere());
	}

	void Gizmos::DrawDisc(const Vector3& center, const Vector3& normal, float radius)
	{
		if (s_Context.event != GizmoEvent::Draw)
		{
			return;
		}
		if (s_ArcCount + 1 >= MAX_ARCS)
		{
			FlushArcs();
		}
		Vector3 tangent = normal.Cross(Vector3::Up);
		if (tangent.LengthSquared() < 0.001f)
		{
			tangent = normal.Cross(Vector3::Right);
		}
		s_Arcs[s_ArcCount++] = { center, normal, radius, tangent, 360, s_Context.color };
	}

	void Gizmos::DrawFrustum(const Frustum& frustum)
	{
		if (s_Context.event != GizmoEvent::Draw)
		{
			return;
		}
		if (s_LineCount + 12 >= MAX_LINES)
		{
			FlushLines();
		}
		Vector3 corners[8];
		frustum.GetCorners(corners);

		s_Lines[s_LineCount++] = { corners[0], s_Context.color, corners[1], s_Context.color };
		s_Lines[s_LineCount++] = { corners[1], s_Context.color, corners[2], s_Context.color };
		s_Lines[s_LineCount++] = { corners[2], s_Context.color, corners[3], s_Context.color };
		s_Lines[s_LineCount++] = { corners[3], s_Context.color, corners[0], s_Context.color };

		s_Lines[s_LineCount++] = { corners[4], s_Context.color, corners[5], s_Context.color };
		s_Lines[s_LineCount++] = { corners[5], s_Context.color, corners[6], s_Context.color };
		s_Lines[s_LineCount++] = { corners[6], s_Context.color, corners[7], s_Context.color };
		s_Lines[s_LineCount++] = { corners[7], s_Context.color, corners[4], s_Context.color };

		s_Lines[s_LineCount++] = { corners[0], s_Context.color, corners[4], s_Context.color };
		s_Lines[s_LineCount++] = { corners[1], s_Context.color, corners[5], s_Context.color };
		s_Lines[s_LineCount++] = { corners[2], s_Context.color, corners[6], s_Context.color };
		s_Lines[s_LineCount++] = { corners[3], s_Context.color, corners[7], s_Context.color };
	}

	void Gizmos::DrawMesh(const Vector3& position, const Quaternion& rotation, const Vector3& scale, Mesh* mesh)
	{
		DrawMesh(position, rotation, scale, mesh->GetVertexBuffer(), mesh->GetIndexBuffer());
	}

	void Gizmos::DrawMesh(const Vector3& position, const Quaternion& rotation, const Vector3& scale, GfxBuffer* vertexBuffer, GfxBuffer* indexBuffer)
	{
		if (s_Context.event != GizmoEvent::Draw)
		{
			return;
		}
		BindColorAndThickness(s_Context.color, s_Context.thickness);
		BindMatrix(Math::CreateTRS(position, rotation, scale) * s_Context.matrix);
		uint32_t indexCount = indexBuffer == nullptr ? 0 : indexBuffer->GetElementCount();
		if (s_Context.ignoreDepth)
		{
			GfxDevice::Draw(GfxDrawingOperation(vertexBuffer, indexBuffer, s_GizmosMaterial, &s_MeshLayout, indexCount, 0, vertexBuffer->GetElementCount(), Topology::TriangleList, 8, nullptr, 0, 1, false, false));
		}
		else
		{
			GfxDevice::Draw(GfxDrawingOperation(vertexBuffer, indexBuffer, s_GizmosMaterial, &s_MeshLayout, indexCount, 0, vertexBuffer->GetElementCount(), Topology::TriangleList, 6, nullptr, 0, 1, false, false));
			GfxDevice::Draw(GfxDrawingOperation(vertexBuffer, indexBuffer, s_GizmosMaterial, &s_MeshLayout, indexCount, 0, vertexBuffer->GetElementCount(), Topology::TriangleList, 7, nullptr, 0, 1, false, false));
		}
	}

	void Gizmos::DrawSolidMesh(const Vector3& position, const Quaternion& rotation, const Vector3& scale, Mesh* mesh)
	{
		DrawSolidMesh(position, rotation, scale, mesh->GetVertexBuffer(), mesh->GetIndexBuffer());
	}

	void Gizmos::DrawSolidMesh(const Vector3& position, const Quaternion& rotation, const Vector3& scale, GfxBuffer* vertexBuffer, GfxBuffer* indexBuffer)
	{
		if (s_Context.event != GizmoEvent::Draw)
		{
			return;
		}
		BindColorAndThickness(s_Context.color, s_Context.thickness);
		BindMatrix(Math::CreateTRS(position, rotation, scale) * s_Context.matrix);
		uint32_t indexCount = indexBuffer == nullptr ? 0 : indexBuffer->GetElementCount();
		if (s_Context.ignoreDepth)
		{
			GfxDevice::Draw(GfxDrawingOperation(vertexBuffer, indexBuffer, s_GizmosMaterial, &s_MeshLayout, indexCount, 0, vertexBuffer->GetElementCount(), Topology::TriangleList, 8, nullptr, 0, 1, false, true));
		}
		else
		{
			GfxDevice::Draw(GfxDrawingOperation(vertexBuffer, indexBuffer, s_GizmosMaterial, &s_MeshLayout, indexCount, 0, vertexBuffer->GetElementCount(), Topology::TriangleList, 6, nullptr, 0, 1, false, true));
			GfxDevice::Draw(GfxDrawingOperation(vertexBuffer, indexBuffer, s_GizmosMaterial, &s_MeshLayout, indexCount, 0, vertexBuffer->GetElementCount(), Topology::TriangleList, 7, nullptr, 0, 1, false, true));
		}
	}

	void Gizmos::FlushLines()
	{
		if (s_LineCount == 0)
		{
			return;
		}

		s_LineVertexBuffer->SetData(s_Lines.data(), s_LineCount * sizeof(Line));

		BindColorAndThickness(s_Context.color, s_Context.thickness);
		BindMatrix(s_Context.matrix);
		if (s_Context.ignoreDepth)
		{
			if (s_Context.thickness > 0)
			{
				GfxDevice::Draw(GfxDrawingOperation(s_LineVertexBuffer, nullptr, s_GizmosMaterial, &s_LineLayout, 0, 0, s_LineCount * 2, Topology::LineList, 9));
			}
			else
			{
				GfxDevice::Draw(GfxDrawingOperation(s_LineVertexBuffer, nullptr, s_GizmosMaterial, &s_LineLayout, 0, 0, s_LineCount * 2, Topology::LineList, 2));
			}
		}
		else
		{
			GfxDevice::Draw(GfxDrawingOperation(s_LineVertexBuffer, nullptr, s_GizmosMaterial, &s_LineLayout, 0, 0, s_LineCount * 2, Topology::LineList, 0));
			GfxDevice::Draw(GfxDrawingOperation(s_LineVertexBuffer, nullptr, s_GizmosMaterial, &s_LineLayout, 0, 0, s_LineCount * 2, Topology::LineList, 1));
		}
		
		s_LineCount = 0;
	}

	void Gizmos::FlushArcs()
	{
		if (s_ArcCount == 0)
		{
			return;
		}

		s_ArcVertexBuffer->SetData(s_Arcs.data(), s_ArcCount * sizeof(Arc));

		BindColorAndThickness(s_Context.color, s_Context.thickness);
		BindMatrix(s_Context.matrix);
		if (s_Context.ignoreDepth)
		{
			if (s_Context.thickness > 0)
			{
				GfxDevice::Draw(GfxDrawingOperation(s_ArcVertexBuffer, nullptr, s_GizmosMaterial, &s_ArcLayout, 0, 0, s_ArcCount, Topology::PointList, 10));
			}
			else
			{
				GfxDevice::Draw(GfxDrawingOperation(s_ArcVertexBuffer, nullptr, s_GizmosMaterial, &s_ArcLayout, 0, 0, s_ArcCount, Topology::PointList, 5));
			}
		}
		else
		{
			GfxDevice::Draw(GfxDrawingOperation(s_ArcVertexBuffer, nullptr, s_GizmosMaterial, &s_ArcLayout, 0, 0, s_ArcCount, Topology::PointList, 3));
			GfxDevice::Draw(GfxDrawingOperation(s_ArcVertexBuffer, nullptr, s_GizmosMaterial, &s_ArcLayout, 0, 0, s_ArcCount, Topology::PointList, 4));
		}
	
		s_ArcCount = 0;
	}

	void Gizmos::FlushTriangles()
	{
		if (s_TriangleCount == 0)
		{
			return;
		}

		s_TriangleVertexBuffer->SetData(s_Triangles.data(), s_TriangleCount * sizeof(Triangle));

		BindColorAndThickness(s_Context.color, s_Context.thickness);
		BindMatrix(s_Context.matrix);
		if (s_Context.ignoreDepth)
		{
			GfxDevice::Draw(GfxDrawingOperation(s_TriangleVertexBuffer, nullptr, s_GizmosMaterial, &s_TriangleLayout, 0, 0, s_TriangleCount * 3, Topology::TriangleList, 2));
		}
		else
		{
			GfxDevice::Draw(GfxDrawingOperation(s_TriangleVertexBuffer, nullptr, s_GizmosMaterial, &s_TriangleLayout, 0, 0, s_TriangleCount * 3, Topology::TriangleList, 0));
			GfxDevice::Draw(GfxDrawingOperation(s_TriangleVertexBuffer, nullptr, s_GizmosMaterial, &s_TriangleLayout, 0, 0, s_TriangleCount * 3, Topology::TriangleList, 1));
		}

		s_TriangleCount = 0;
	}

	void Gizmos::BindMatrix(const Matrix& matrix)
	{
		if (matrix == s_Context.bindedMatrix)
		{
			return;
		}
		s_Context.bindedMatrix = matrix;
		PerDrawDataConstantBuffer::BindData(matrix);
	}

	void Gizmos::BindColorAndThickness(const Color& color, float thickness)
	{
		if (color == s_Context.bindedColor && thickness == s_Context.bindedThickness)
		{
			return;
		}
		s_Context.bindedColor = color;
		s_Context.bindedThickness = thickness;
		GizmoDataConstantBuffer::BindData(color, thickness);
	}

	// Handles

	bool Gizmos::IsInUse()
	{
		return s_Context.closestHandle != UINT32_MAX || s_Context.activeHandle != UINT32_MAX;
	}

	constexpr uint32_t GetStrId(const char* str)
	{
		size_t length = 0;
		while (str[length] != '\0')
		{
			++length;
		}

		return CRCHelper::Calculate(str, length, 0);
	}

	void Gizmos::PushId(uint32_t id)
	{
		s_Context.idStack.push_back(GetId(id));
	}

	void Gizmos::PopId()
	{
		s_Context.idStack.pop_back();
	}

	float PointOnLineParameter(const Vector3& point, const Vector3& linePoint, const Vector3& lineDirection)
	{
		return (lineDirection.Dot(point - linePoint)) / lineDirection.LengthSquared();
	}

	bool CalcParamOnConstraint(Camera* camera, const Vector2& mousePosition, const Vector3& constraintOrigin, const Vector3& constraintDir, float& parameterization)
	{
		Vector3 constraintToCameraTangent = constraintDir.Cross(camera->GetTransform()->GetPosition() - constraintOrigin);
		Vector3 constraintPlaneNormal = constraintDir.Cross(constraintToCameraTangent);
		Plane plane = Plane(constraintOrigin, constraintPlaneNormal);
		Ray ray = SceneHelper::ScreenPointToWorldRay(mousePosition, camera);
		float distance;
		if (ray.direction.Dot(plane.Normal()) > 0.05f && Math::Raycast(plane, ray, distance))
		{
			Vector3 pointOnPlane = ray.position + ray.direction * distance;
			parameterization = PointOnLineParameter(pointOnPlane, constraintOrigin, constraintDir);
			return !std::isinf(parameterization);
		}
		parameterization = 0.0f;
		return false;
	}

	bool CalcPositionOnConstraint(Camera* camera, const Vector2& mousePosition, const Vector3& constraintOrigin, const Vector3& constraintDir, Vector3& position)
	{
		float pointOnLineParam;
		if (CalcParamOnConstraint(camera, mousePosition, constraintOrigin, constraintDir, pointOnLineParam))
		{
			position = constraintOrigin + constraintDir * pointOnLineParam;
			return true;
		}
		position = Vector3::Zero;
		return false;
	}

	Vector2 WorldToScreenPoint(const Vector3& position)
	{
		Vector3 transformedPosition = Math::MultiplyPoint(s_Context.matrix, position);
		Vector3 screenPosition = s_Context.camera->WorldToScreenPoint(transformedPosition);
		screenPosition.y = s_Context.camera->GetPixelSize().y - screenPosition.y;
		return Vector2(screenPosition);
	}

	Vector2 ProjectPointLine(const Vector2& position, const Vector2& from, const Vector2& to)
	{
		Vector2 relativePoint = position - from;
		Vector2 lineDirection = to - from;
		float length = lineDirection.Length();
		Vector2 normalizedLineDirection = lineDirection;
		if (length > 0.000001f)
		{
			normalizedLineDirection /= length;
		}
		float dot = normalizedLineDirection.Dot(relativePoint);
		dot = std::clamp(dot, 0.0f, length);
		return from + normalizedLineDirection * dot;
	}

	Vector3 ProjectPointLine(const Vector3& position, const Vector3& from, const Vector3& to)
	{
		Vector3 relativePoint = position - from;
		Vector3 lineDirection = to - from;
		float length = lineDirection.Length();
		Vector3 normalizedLineDirection = lineDirection;
		if (length > 0.000001f)
		{
			normalizedLineDirection /= length;
		}
		float dot = normalizedLineDirection.Dot(relativePoint);
		dot = std::clamp(dot, 0.0f, length);
		return from + normalizedLineDirection * dot;
	}

	float DistanceToLine(const Vector2& screenFrom, const Vector2& screenTo)
	{
		Vector2 screenPosition = s_Context.mousePosition;
		float distance = (ProjectPointLine(screenPosition, screenFrom, screenTo) - screenPosition).Length();
		return std::max(0.0f, distance);
	}

	float DistanceToLine(const Vector3& from, const Vector3& to)
	{
		Vector2 screenPosition = s_Context.mousePosition;
		Vector2 screenFrom = WorldToScreenPoint(from);
		Vector2 screenTo = WorldToScreenPoint(to);
		float distance = (ProjectPointLine(screenPosition, screenFrom, screenTo) - screenPosition).Length();
		return std::max(0.0f, distance);
	}

	float DistanceToCircle(const Vector3& position, const Quaternion& rotation, float size)
	{
		Vector2 screenCenter = WorldToScreenPoint(position);
		Vector2 screenEdge = WorldToScreenPoint(position + Math::MultiplyPoint(s_Context.camera->GetTransform()->GetRotation(), Vector3::Right) * size);
		size = (screenCenter - screenEdge).Length();
		float distance = (screenCenter - s_Context.mousePosition).Length();
		if (distance < size)
		{
			return 0;
		}
		return distance - size;
	}

	float DistanceToPolyLine(Vector3* points, uint32_t count)
	{
		Vector2 screenPosition = s_Context.mousePosition;
		Vector2 p1 = WorldToScreenPoint(points[0]);
		Vector2 p2 = WorldToScreenPoint(points[1]);
		float distance = DistanceToLine(p1, p2);
		for (uint32_t i = 2; i < count; i++)
		{
			p1 = p2;
			p2 = WorldToScreenPoint(points[i]);
			float newDistance = DistanceToLine(p1, p2);
			if (newDistance < distance)
			{
				distance = newDistance;
			}
		}
		return distance;
	}

	float DistanceToArc(const Vector3& center, const Vector3& normal, const Vector3& from, float angle, float radius)
	{
		Vector3 bitangent = normal.Cross(from);
		Vector3 points[37];
		float segments = 36.0;
		for (int i = 0; i <= 36; ++i)
		{
			float u = cos(Math::ToRadians(i / segments * angle)) * radius;
			float v = sin(Math::ToRadians(i / segments * angle)) * radius;
			points[i] = center + from * u + bitangent * v;
		}
		return DistanceToPolyLine(points, 37);
	}

	float DistanceToDisk(const Vector3& center, const Vector3& normal, float radius)
	{
		Vector3 tangent = normal.Cross(Vector3::Up);
		if (tangent.LengthSquared() < 0.001f)
		{
			tangent = normal.Cross(Vector3::Right);
		}
		return DistanceToArc(center, normal, tangent, 360, radius);
	}

	float DistanceToArrow(const Vector3& position, const Quaternion& rotation, float size)
	{
		Vector3 arrowPosition = position + Math::MultiplyPoint(rotation, Vector3(0.0f, 0.0f, s_TransformGizmoScale * size));
		return std::fminf(DistanceToLine(position, arrowPosition), DistanceToCircle(arrowPosition, rotation, size * 0.15f));
	}

	bool IsAxisActive(uint32_t id)
	{
		for (uint32_t i = 0; i < ARRAYSIZE(s_AxisHandleId); ++i)
		{
			if (s_AxisHandleId[i] == id)
			{
				return true;
			}
		}
		return false;
	}

	Color GetPositionCapColor(uint32_t id)
	{
		if (s_Context.activeHandle == id || (!s_Context.isMouseDown && s_Context.closestHandle == id))
		{
			return Color(1, 0.5f, 0.0f, 1.0f);
		}
		else if (s_Context.isMouseDown && s_Context.activeHandle != UINT32_MAX && s_Context.activeHandle != id && IsAxisActive(s_Context.activeHandle))
		{
			return Gizmos::GetColor(0.15f);
		}
		else
		{
			return Gizmos::GetColor(1.0f);
		}
	}

	Color GetRotationCapColor(uint32_t id)
	{
		if (s_Context.activeHandle == id || (!s_Context.isMouseDown && s_Context.closestHandle == id))
		{
			return Color(1, 0.5f, 0.0f, 1.0f);
		}
		else if (s_Context.isMouseDown && s_Context.activeHandle != UINT32_MAX && s_Context.activeHandle != id && IsAxisActive(s_Context.activeHandle))
		{
			return Color(0, 0, 0, 0);
		}
		else
		{
			return Gizmos::GetColor(1.0f);
		}
	}

	void CalculateAxisDrawOrder(const Vector3& position, const Vector3 direction[3], int order[3])
	{
		Vector3 worldPosition = Math::MultiplyPoint(s_Context.matrix, position);
		Vector3 viewDirection = s_Context.camera->IsOrthographic() ? -s_Context.camera->GetInverseViewMatrix().Forward() : worldPosition - s_Context.camera->GetTransform()->GetPosition();
		Vector3 axisDepth(Math::MultiplyVector(s_Context.matrix, direction[0]).Dot(viewDirection), Math::MultiplyVector(s_Context.matrix, direction[1]).Dot(viewDirection), Math::MultiplyVector(s_Context.matrix, direction[2]).Dot(viewDirection));

		order[0] = 0;
		order[1] = 1;
		order[2] = 2;

		if (axisDepth.y > axisDepth.x)
		{
			std::swap(axisDepth.y, axisDepth.x);
			std::swap(order[1], order[0]);
		}

		if (axisDepth.z > axisDepth.y)
		{
			std::swap(axisDepth.z, axisDepth.y);
			std::swap(order[2], order[1]);
		}

		if (axisDepth.y > axisDepth.x)
		{
			std::swap(axisDepth.y, axisDepth.x);
			std::swap(order[1], order[0]);
		}
	}

	void DrawSquareCap(uint32_t id, const Vector3& position, const Quaternion& rotation, float size)
	{
		Quaternion cameraRotation = s_Context.camera->GetTransform()->GetRotation();
		Vector3 cameraRight = Math::MultiplyVector(cameraRotation, Vector3::Right);
		Vector3 cameraUp = Math::MultiplyVector(cameraRotation, Vector3::Up);
		Vector3 right = Math::MultiplyVector(s_Context.inverseMatrix, cameraRight) * size;
		Vector3 up = Math::MultiplyVector(s_Context.inverseMatrix, cameraUp) * size;

		bool currentIsIgnoringDepth = Gizmos::GetIgnoreDepth();
		Color currentColor = Gizmos::GetColor();
		Color color = s_Context.activeHandle == id ? Gizmos::GetColor(1.0f) : (s_Context.closestHandle == id ? Gizmos::GetColor(0.9f) : Gizmos::GetColor(0.8f));
		Gizmos::SetIgnoreDepth(true);
		Gizmos::SetColor(color);
		Gizmos::DrawTriangle(position + right + up, position + right - up, position - right - up);
		Gizmos::DrawTriangle(position - right - up, position - right + up, position + right + up);
		Gizmos::SetIgnoreDepth(currentIsIgnoringDepth);
		Gizmos::SetColor(currentColor);
	}

	void DrawPositionSquareCap(uint32_t id, const Vector3& position, const Quaternion& rotation, float size)
	{
		Vector3 sideways = Math::MultiplyPoint(rotation, Vector3(size, 0, 0));
		Vector3 up = Math::MultiplyPoint(rotation, Vector3(0, size, 0));
		bool currentIsIgnoringDepth = Gizmos::GetIgnoreDepth();
		Color borderColor = GetPositionCapColor(id);
		Color solidColor = borderColor;
		solidColor.w *= 0.3f;
		Color currentColor = Gizmos::GetColor();
		Vector3 p0 = position + sideways + up;
		Vector3 p1 = position + sideways - up;
		Vector3 p2 = position - sideways - up;
		Vector3 p3 = position - sideways + up;
		Gizmos::SetIgnoreDepth(true);
		Gizmos::SetColor(solidColor);
		Gizmos::DrawTriangle(p0, p1, p2);
		Gizmos::DrawTriangle(p2, p3, p0);
		Gizmos::SetColor(borderColor);
		Gizmos::DrawLine(p0, p1);
		Gizmos::DrawLine(p1, p2);
		Gizmos::DrawLine(p2, p3);
		Gizmos::DrawLine(p3, p0);
		Gizmos::SetIgnoreDepth(currentIsIgnoringDepth);
		Gizmos::SetColor(currentColor);
	}

	void DrawPositionCap(uint32_t id, const Vector3& position, const Quaternion& rotation, float size)
	{
		Vector3 arrowPosition = position + Math::MultiplyPoint(rotation, Vector3(0.0f, 0.0f, s_TransformGizmoScale * size));
		Quaternion arrowRotation = Quaternion::CreateFromYawPitchRoll(0, Math::ToRadians(90), 0) * rotation;
		bool currentIsIgnoringDepth = Gizmos::GetIgnoreDepth();
		Color currentColor = Gizmos::GetColor();
		float currentThickness = Gizmos::GetThickness();
		Gizmos::SetIgnoreDepth(true);
		Gizmos::SetColor(GetPositionCapColor(id));
		Gizmos::SetThickness(3.0f);
		Gizmos::DrawLine(position, arrowPosition);
		Gizmos::DrawSolidMesh(arrowPosition, arrowRotation, Vector3::One * size * 0.15f, StandardMeshes::GetCone());
		Gizmos::SetIgnoreDepth(currentIsIgnoringDepth);
		Gizmos::SetColor(currentColor);
		Gizmos::SetThickness(0.0f);
	}

	void DrawScaleCap(uint32_t id, const Vector3& position, const Quaternion& rotation, float size)
	{
		Vector3 cubePosition = position + Math::MultiplyPoint(rotation, Vector3(0.0f, 0.0f, s_TransformGizmoScale * size));
		Quaternion arrowRotation = Quaternion::CreateFromYawPitchRoll(0, Math::ToRadians(90), 0) * rotation;
		bool currentIsIgnoringDepth = Gizmos::GetIgnoreDepth();
		Color currentColor = Gizmos::GetColor();
		float currentThickness = Gizmos::GetThickness();
		Gizmos::SetIgnoreDepth(true);
		Gizmos::SetColor(GetPositionCapColor(id));
		Gizmos::SetThickness(3.0f);
		Gizmos::DrawLine(position, cubePosition);
		Gizmos::DrawSolidMesh(cubePosition, arrowRotation, Vector3::One * size * 0.15f, StandardMeshes::GetCube());
		Gizmos::SetIgnoreDepth(currentIsIgnoringDepth);
		Gizmos::SetColor(currentColor);
		Gizmos::SetThickness(0.0f);
	}

	bool BoxBoundsSlider(uint32_t id, Vector3& position, Vector3& size, const Vector3& direction)
	{
		Vector3 axisMask = Vector3(std::abs(direction.x), std::abs(direction.y), std::abs(direction.z));
		float originalValue = axisMask.Dot(size * 0.5f);
		Vector3 handlePosition = position + direction * originalValue;
		float handleSize = Gizmos::GetHandleSize(handlePosition);

		if (Gizmos::Slider1D(id, handlePosition, direction, handleSize * 0.04f, &DistanceToCircle, &DrawSquareCap))
		{
			float newValue = (handlePosition - position).Dot(direction);
			float delta = newValue - originalValue;
			size += axisMask * delta;
			position += direction * delta * 0.5f;
			return true;
		}
		return false;
	}

	bool SphereBoundsSlider(uint32_t id, Vector3& position, float& radius, const Vector3& direction)
	{
		Vector3 handlePosition = position + direction * radius;
		float handleSize = Gizmos::GetHandleSize(handlePosition);

		if (Gizmos::Slider1D(id, handlePosition, direction, handleSize * 0.04f, &DistanceToCircle, &DrawSquareCap))
		{
			float newValue = (handlePosition - position).Dot(direction);
			float delta = newValue - radius;
			position += direction * delta * 0.5f;
			radius += delta * 0.5f;
			return true;
		}
		return false;
	}

	bool TryGetRotationPlaneVector(const Vector3& center, const Vector3& axis, Vector3& vector)
	{
		Ray ray = SceneHelper::ScreenPointToWorldRay(s_Context.mousePosition, s_Context.camera);
		float denominator = ray.direction.Dot(axis);
		float t = (center - ray.position).Dot(axis) / denominator;
		Vector3 point = ray.position + ray.direction * t;
		vector = point - center;
		vector -= axis * vector.Dot(axis);
		vector.Normalize();
		return true;
	}

	bool RotationHandleArc(uint32_t id, const Vector3& position, Quaternion& rotation, const Vector3& normal, const Vector3& from, float angle, float size)
	{
		static Vector3 dragCenter;
		static Vector3 dragAxisWorld;
		static Vector3 dragAxisLocal;
		static Vector3 previousVector;
		static Vector3 dragStartVectorLocal;
		static Quaternion startRotation;
		static float totalAngle = 0.0f;

		bool result = false;
		switch (s_Context.event)
		{
		case GizmoEvent::Search:
			Gizmos::TrySetClosestHandle(id, DistanceToArc(position, normal, from, angle, size));
			break;
		case GizmoEvent::Input:
			if (s_Context.isMousePressed && s_Context.closestHandle == id)
			{
				dragCenter = Math::MultiplyPoint(s_Context.matrix, position);
				dragAxisWorld = Math::MultiplyVector(s_Context.matrix, normal);
				dragAxisWorld.Normalize();

				dragAxisLocal = normal;
				dragAxisLocal.Normalize();

				if (TryGetRotationPlaneVector(dragCenter, dragAxisWorld, previousVector))
				{
					dragStartVectorLocal = Math::MultiplyVector(s_Context.inverseMatrix, previousVector);
					dragStartVectorLocal -= dragAxisLocal * dragStartVectorLocal.Dot(dragAxisLocal);
					dragStartVectorLocal.Normalize();

					startRotation = rotation;
					totalAngle = 0.0f;
					s_Context.activeHandle = id;
				}
			}
			if (s_Context.isMouseDragging && s_Context.activeHandle == id)
			{
				Vector3 currentVector;
				if (TryGetRotationPlaneVector(dragCenter, dragAxisWorld, currentVector))
				{
					float step = std::atan2(dragAxisWorld.Dot(previousVector.Cross(currentVector)), previousVector.Dot(currentVector));
					totalAngle += step;
					previousVector = currentVector;
					Quaternion deltaLocal = Quaternion::CreateFromAxisAngle(dragAxisLocal, totalAngle);
					rotation = startRotation * deltaLocal;
					result = true;
				}
			}
			break;
		case GizmoEvent::Draw:
			Color currentColor = Gizmos::GetColor();
			Gizmos::SetColor(GetRotationCapColor(id));
			if (s_Context.activeHandle == id)
			{
				Color currentColor = Gizmos::GetColor();
				Gizmos::SetColor(Gizmos::GetColor(0.25f));
				Gizmos::DrawSolidArc(position, dragAxisLocal, dragStartVectorLocal, Math::ToDegrees(totalAngle), size);
				Gizmos::SetColor(currentColor);

				Gizmos::DrawDisc(position, normal, size);

				Vector3 bitangent = dragAxisLocal.Cross(dragStartVectorLocal);
				Vector3 startPoint = position + dragStartVectorLocal * size;
				Vector3 endPoint = position + (dragStartVectorLocal * cosf(totalAngle) + bitangent * sinf(totalAngle)) * size;
				Gizmos::DrawLine(position, startPoint);
				Gizmos::DrawLine(position, endPoint);
			}
			else
			{
				Gizmos::DrawArc(position, normal, from, angle, size);
			}
			Gizmos::SetColor(currentColor);
			break;
		}
		return result;
	}

	bool RotationHandleDisc(uint32_t id, const Vector3& position, Quaternion& rotation, const Vector3& normal, float size)
	{
		Vector3 tangent = normal.Cross(Vector3::Up);
		if (tangent.LengthSquared() < 0.001f)
		{
			tangent = normal.Cross(Vector3::Right);
		}
		tangent.Normalize();
		return RotationHandleArc(id, position, rotation, normal, tangent, 360.0f, size);
	}

	bool Gizmos::Slider1D(Vector3& position, const Vector3& direction)
	{
		return Slider1D(GetUniqueId(), position, direction, GetHandleSize(position) * 0.04f, &DistanceToCircle, &DrawSquareCap);
	}

	bool Gizmos::Slider1D(const Vector3& position, const Vector3& direction, float& value)
	{
		Vector3 pos = position + direction * value;
		float handleSize = GetHandleSize(pos);
		if (Slider1D(GetUniqueId(), pos, direction, handleSize * 0.04f, &DistanceToCircle, &DrawSquareCap))
		{
			value = (pos - position).Dot(direction);
			return true;
		}
		return false;
	}

	bool Gizmos::Slider1D(Vector3& position, const Vector3& direction, float size, const HandleDistanceFunction& distanceFunction, const HandleDrawFunction& drawFunction)
	{
		return Slider1D(GetUniqueId(), position, direction, size, distanceFunction, drawFunction);
	}

	bool Gizmos::Slider1D(uint32_t id, Vector3& position, const Vector3& direction, float size, const HandleDistanceFunction& distanceFunction, const HandleDrawFunction& drawFunction)
	{
		static Vector3 constraintOrigin;
		static Vector3 constraintDir;
		static Vector3 startHandleOffset;
		static float startHandleSize;

		bool result = false;
		switch (s_Context.event)
		{
		case GizmoEvent::Search:
			TrySetClosestHandle(id, distanceFunction(position, Math::LookRotation(direction), size));
			break;
		case GizmoEvent::Input:
			if (s_Context.isMousePressed && s_Context.closestHandle == id)
			{
				constraintOrigin = Math::MultiplyPoint(s_Context.matrix, position);
				constraintDir = Math::MultiplyVector(s_Context.matrix, direction);
				Vector3 point;
				startHandleOffset = CalcPositionOnConstraint(s_Context.camera, s_Context.mousePosition, constraintOrigin, constraintDir, point) ? constraintOrigin - point : Vector3::Zero;
				startHandleSize = GetHandleSize(point);
				s_Context.activeHandle = id;
			}
			if (s_Context.isMouseDragging && s_Context.activeHandle == id)
			{
				Vector3 worldPosition;
				if (CalcPositionOnConstraint(s_Context.camera, s_Context.mousePosition, constraintOrigin, constraintDir, worldPosition))
				{
					Vector3 handleOffset = startHandleOffset * (GetHandleSize(worldPosition) / startHandleSize);
					worldPosition += handleOffset;
					position = Math::MultiplyPoint(s_Context.inverseMatrix, worldPosition);
					result = true;
				}
			}
			break;
		case GizmoEvent::Draw:
			drawFunction(id, position, Math::LookRotation(direction), size);
			break;
		}
		return result;
	}

	bool Gizmos::Slider2D(uint32_t id, Vector3& handlePosition, const Vector3& handleDirection, const Vector3& slideDirection1, const Vector3& slideDirection2, float size, const HandleDistanceFunction& distanceFunction, const HandleDrawFunction& drawFunction)
	{
		auto GetMousePositionOnPlane = [](const Vector3& position, const Vector3& direction, Vector3& result) -> bool
		{
			Plane plane = Plane(Math::MultiplyPoint(s_Context.matrix, position), Math::MultiplyVector(s_Context.matrix, direction));
			Ray ray = SceneHelper::ScreenPointToWorldRay(s_Context.mousePosition, s_Context.camera);
			float distance;
			if (Math::Raycast(plane, ray, distance))
			{
				result = ray.position + ray.direction * distance;
				return true;
			}
			return false;
		};

		static Vector3 startPosition;
		static Vector2 startPlaneOffset;

		Quaternion handleRotation = Math::LookRotation(handleDirection, slideDirection1);

		bool result = false;
		switch (s_Context.event)
		{
		case GizmoEvent::Search:
			TrySetClosestHandle(id, distanceFunction(handlePosition, handleRotation, size));
			break;
		case GizmoEvent::Input:
			if (s_Context.isMousePressed && s_Context.closestHandle == id)
			{
				Vector3 worldMousePosition;
				if (GetMousePositionOnPlane(handlePosition, handleDirection, worldMousePosition))
				{
					Vector3 localMousePosition = Math::MultiplyPoint(s_Context.inverseMatrix, worldMousePosition);
					startPosition = handlePosition;
					Vector3 clickOffset = localMousePosition - handlePosition;
					startPlaneOffset.x = clickOffset.Dot(slideDirection1);
					startPlaneOffset.y = clickOffset.Dot(slideDirection2);
					s_Context.activeHandle = id;
				}
			}
			if (s_Context.isMouseDragging && s_Context.activeHandle == id)
			{
				Vector3 worldMousePosition;
				if (GetMousePositionOnPlane(handlePosition, handleDirection, worldMousePosition))
				{
					Vector3 localMousePosition = Math::MultiplyPoint(s_Context.inverseMatrix, worldMousePosition);
					Vector2 delta = Vector2(PointOnLineParameter(localMousePosition, startPosition, slideDirection1), PointOnLineParameter(localMousePosition, startPosition, slideDirection2));
					delta -= startPlaneOffset;
					handlePosition = startPosition + slideDirection1 * delta.x + slideDirection2 * delta.y;
					result = true;
				}
			}
			break;
		case GizmoEvent::Draw:
			drawFunction(id, handlePosition, handleRotation, size);
			break;
		}
		return result;
	}

	bool Gizmos::BoxBoundsHandle(Vector3& position, Vector3& size)
	{
		PushId(GetUniqueId());
		if (s_Context.event == GizmoEvent::Draw)
		{
			Color previousColor = Gizmos::GetColor();
			SetColor(GetColor(0.125f));
			DrawSolidBox(position, size);
			SetColor(previousColor);
		}

		bool result = false;
		result |= BoxBoundsSlider(GetStrId("RightSizeSlider"), position, size, Vector3::Right);
		result |= BoxBoundsSlider(GetStrId("LeftSizeSlider"), position, size, Vector3::Left);
		result |= BoxBoundsSlider(GetStrId("UpSizeSlider"), position, size, Vector3::Up);
		result |= BoxBoundsSlider(GetStrId("DownSizeSlider"), position, size, Vector3::Down);
		result |= BoxBoundsSlider(GetStrId("ForwardSizeSlider"), position, size, Vector3::Forward);
		result |= BoxBoundsSlider(GetStrId("BackwardSizeSlider"), position, size, Vector3::Backward);
		PopId();
		return result;
	}

	bool Gizmos::SphereBoundsHandle(Vector3& position, float& radius)
	{
		PushId(GetUniqueId());
		if (s_Context.event == GizmoEvent::Draw)
		{
			Color previousColor = Gizmos::GetColor();
			SetColor(GetColor(0.125f));
			DrawSolidSphere(position, radius);
			SetColor(previousColor);
		}

		bool result = false;
		result |= SphereBoundsSlider(GetStrId("RightSizeSlider"), position, radius, Vector3::Right);
		result |= SphereBoundsSlider(GetStrId("LeftSizeSlider"), position, radius, Vector3::Left);
		result |= SphereBoundsSlider(GetStrId("UpSizeSlider"), position, radius, Vector3::Up);
		result |= SphereBoundsSlider(GetStrId("DownSizeSlider"), position, radius, Vector3::Down);
		result |= SphereBoundsSlider(GetStrId("ForwardSizeSlider"), position, radius, Vector3::Forward);
		result |= SphereBoundsSlider(GetStrId("BackwardSizeSlider"), position, radius, Vector3::Backward);
		PopId();
		return result;
	}

	bool Gizmos::PositionHandle(Vector3& position, const Quaternion& rotation, bool isLocal)
	{
		Quaternion orientation = isLocal ? rotation : Quaternion::Identity;
		Vector3 axis[3] = { Math::MultiplyVector(orientation, s_AxisHandleDirection[0]), Math::MultiplyVector(orientation, s_AxisHandleDirection[1]), Math::MultiplyVector(orientation, s_AxisHandleDirection[2]) };

		int order[3];
		CalculateAxisDrawOrder(position, axis, order);

		Vector3 worldPosition = Math::MultiplyPoint(s_Context.matrix, position);
		Vector3 inverseViewDirection = s_Context.camera->IsOrthographic() ? s_Context.camera->GetInverseViewMatrix().Forward() : s_Context.camera->GetTransform()->GetPosition() - worldPosition;
		
		PushId(GetUniqueId());
		bool result = false;
		float handleSize = GetHandleSize(position);
		Color currentColor = GetColor();
		for (int i = 0; i < 3; ++i)
		{
			int j = ((order[i] + 2) % 3);
			Vector3 planeDirection = Math::MultiplyVector(orientation, s_PlaneHandleDirection[j]);
			if (s_Context.activeHandle == s_AxisHandleId[j + 3] || std::abs(s_PlaneHandleDirection[j].Dot(inverseViewDirection)) >= 0.5f)
			{
				SetColor(s_AxisHandleColor[j + 3]);
				Vector3 planeOffset = Math::MultiplyVector(orientation, s_PlaneOffset[j]) * handleSize * 0.35f;
				Vector3 planePosition = position + planeOffset;
				Vector3 slideDirection1 = Math::MultiplyVector(orientation, s_PlaneSlideDirection1[j]);
				Vector3 slideDirection2 = Math::MultiplyVector(orientation, s_PlaneSlideDirection2[j]);
				if (Slider2D(s_AxisHandleId[j + 3], planePosition, planeDirection, slideDirection1, slideDirection2, handleSize * 0.1f, &DistanceToCircle, &DrawPositionSquareCap))
				{
					position = planePosition - planeOffset;
					result = true;
				}
			}
			j = order[i];
			SetColor(s_AxisHandleColor[j]);
			if (Slider1D(s_AxisHandleId[j], position, axis[j], handleSize, &DistanceToArrow, &DrawPositionCap))
			{
				result = true;
			}
		}
		SetColor(currentColor);
		PopId();
		return result;
	}

	bool Gizmos::RotationHandle(const Vector3& position, Quaternion& rotation, bool isLocal)
	{
		Quaternion orientation = isLocal ? rotation : Quaternion::Identity;
		Vector3 axis[3] = { Math::MultiplyVector(orientation, s_AxisHandleDirection[0]), Math::MultiplyVector(orientation, s_AxisHandleDirection[1]), Math::MultiplyVector(orientation, s_AxisHandleDirection[2]) };
		
		int order[3];
		CalculateAxisDrawOrder(position, axis, order);

		Vector3 worldPosition = Math::MultiplyPoint(s_Context.matrix, position);
		Vector3 inverseViewDirection = s_Context.camera->IsOrthographic() ? s_Context.camera->GetInverseViewMatrix().Forward() : s_Context.camera->GetTransform()->GetPosition() - worldPosition;
		inverseViewDirection = Math::MultiplyVector(s_Context.inverseMatrix, inverseViewDirection);

		PushId(GetUniqueId());
		bool result = false;
		float handleSize = GetHandleSize(position);
		bool currentIsIgnoringDepth = Gizmos::GetIgnoreDepth();
		Color currentColor = GetColor();
		float currentThickness = Gizmos::GetThickness();
		SetIgnoreDepth(true);
		SetThickness(3.0f);
		for (int i = 0; i < 3; ++i)
		{
			int j = order[i];
			SetColor(s_AxisHandleColor[j]);

			Vector3 normal = axis[j];
			Vector3 towardCameraInPlane = inverseViewDirection - normal * inverseViewDirection.Dot(normal);
			towardCameraInPlane.Normalize();
			Vector3 from = towardCameraInPlane.Cross(normal);

			if (RotationHandleArc(s_AxisHandleId[j], position, rotation, normal, from, 180.0f, s_TransformGizmoScale * handleSize))
			{
				result = true;
			}
		}
		inverseViewDirection.Normalize();
		SetColor(s_AxisHandleColor[6]);
		if (RotationHandleDisc(s_AxisHandleId[6], position, rotation, inverseViewDirection, s_TransformGizmoScale * handleSize))
		{
			result = true;
		}
		SetIgnoreDepth(currentIsIgnoringDepth);
		SetColor(currentColor);
		SetThickness(0.0f);
		PopId();
		return result;
	}

	bool Gizmos::ScaleHandle(const Vector3& position, const Quaternion& rotation, Vector3& scale, bool isLocal)
	{
		Quaternion orientation = isLocal ? rotation : Quaternion::Identity;
		Vector3 axis[3] = { Math::MultiplyVector(orientation, s_AxisHandleDirection[0]), Math::MultiplyVector(orientation, s_AxisHandleDirection[1]), Math::MultiplyVector(orientation, s_AxisHandleDirection[2]) };
		
		int order[3];
		CalculateAxisDrawOrder(position, axis, order);

		PushId(GetUniqueId());
		bool result = false;
		float handleSize = GetHandleSize(position);
		Color currentColor = GetColor();
		for (int i = 0; i < 3; ++i)
		{
			int j = order[i];
			SetColor(s_AxisHandleColor[j]);

			if (s_Context.event == GizmoEvent::Input && s_Context.isMousePressed && s_Context.closestHandle == s_AxisHandleId[j])
			{
				s_StartScale = scale;
			}

			Vector3 sliderPosition = position;
			if (Slider1D(s_AxisHandleId[j], sliderPosition, axis[j], handleSize, &DistanceToArrow, &DrawScaleCap))
			{
				Vector3 delta = sliderPosition - position;
				Vector3 axisDelta = axis[j] * delta.Dot(axis[j]);
				scale = s_StartScale + axisDelta;
				result = true;
			}
		}
		SetColor(currentColor);
		PopId();
		return result;
	}

	float Gizmos::GetHandleSize(const Vector3& position)
	{
		Vector3 transformedPosition = Math::MultiplyPoint(s_Context.matrix, position);
		Transform* cameraTransform = s_Context.camera->GetTransform();
		Vector3 cameraPosition = cameraTransform->GetPosition();
		Quaternion cameraRotation = cameraTransform->GetRotation();
		float distance = (transformedPosition - cameraPosition).Dot(Math::MultiplyVector(cameraRotation, Vector3::Forward));
		Vector3 screenPos = s_Context.camera->WorldToScreenPoint(cameraPosition + Math::MultiplyVector(cameraRotation, Vector3(0, 0, distance)));
		Vector3 screenPos2 = s_Context.camera->WorldToScreenPoint(cameraPosition + Math::MultiplyVector(cameraRotation, Vector3(1, 0, distance)));
		float screenDist = (screenPos - screenPos2).Length();
		return (80.0f / std::max(screenDist, 0.0001f)) * 1.0f;
	}

	void Gizmos::TrySetClosestHandle(uint32_t id, float distance)
	{
		if (distance <= 5.0f && distance < s_Context.closestDistance)
		{
			s_Context.closestDistance = distance;
			s_Context.closestHandle = id;
		}
	}

	uint32_t Gizmos::GetId(uint32_t id)
	{
		uint32_t seed = s_Context.idStack.empty() ? 0 : s_Context.idStack.back();
		return CRCHelper::Calculate(id, seed);
	}

	uint32_t Gizmos::GetUniqueId()
	{
		return ++s_Context.maxUniqueId;
	}
}
