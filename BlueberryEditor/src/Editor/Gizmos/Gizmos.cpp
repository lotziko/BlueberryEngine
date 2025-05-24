#include "Gizmos.h"

#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\VertexLayout.h"
#include "Blueberry\Graphics\PerDrawDataConstantBuffer.h"

namespace Blueberry
{
	static const uint32_t MAX_LINES = 1024;
	static const uint32_t MAX_VERTICES = MAX_LINES * 2;
	static const uint32_t MAX_INDICES = MAX_LINES * 2;

	static VertexLayout s_LineLayout;
	static VertexLayout s_ArcLayout;

	bool Gizmos::Initialize()
	{
		Shader* lineShader = static_cast<Shader*>(AssetLoader::Load("assets/shaders/Line.shader"));
		if (lineShader == nullptr)
		{
			BB_ERROR("Failed to load gizmo line shader.")
				return false;
		}
		s_LineMaterial = Material::Create(lineShader);

		Shader* arcShader = static_cast<Shader*>(AssetLoader::Load("assets/shaders/ArcLine.shader"));
		if (arcShader == nullptr)
		{
			BB_ERROR("Failed to load gizmo arc shader.")
				return false;
		}
		s_ArcMaterial = Material::Create(arcShader);

		s_LineLayout = VertexLayout{}
			.Append(VertexAttribute::Position, 12)
			.Append(VertexAttribute::Color, 16)
			.Apply();

		int lineSize = s_LineLayout.GetSize();
		s_LineVertexData = new float[MAX_VERTICES * lineSize / sizeof(float)];

		BufferProperties lineBufferProperties = {};
		lineBufferProperties.type = BufferType::Vertex;
		lineBufferProperties.elementCount = MAX_VERTICES;
		lineBufferProperties.elementSize = lineSize;
		lineBufferProperties.isWritable = true;

		if (!GfxDevice::CreateBuffer(lineBufferProperties, s_LineVertexBuffer))
		{
			return false;
		}

		s_Lines = new Line[MAX_LINES];

		s_ArcLayout = VertexLayout{}
			.Append(VertexAttribute::Position, 12)
			.Append(VertexAttribute::Normal, 16)
			.Append(VertexAttribute::Tangent, 16)
			.Append(VertexAttribute::Color, 16)
			.Apply();

		int arcSize = s_ArcLayout.GetSize();
		s_ArcVertexData = new float[MAX_VERTICES * arcSize / sizeof(float)];

		BufferProperties arcBufferProperties = {};
		arcBufferProperties.type = BufferType::Vertex;
		arcBufferProperties.elementCount = MAX_VERTICES;
		arcBufferProperties.elementSize = arcSize;
		arcBufferProperties.isWritable = true;

		if (!GfxDevice::CreateBuffer(arcBufferProperties, s_ArcVertexBuffer))
		{
			return false;
		}

		s_Arcs = new Arc[MAX_LINES];

		return true;
	}

	void Gizmos::Shutdown()
	{
		Material::Destroy(s_LineMaterial);
		delete s_LineVertexData;
		delete s_Lines;
		delete s_LineVertexBuffer;
		Material::Destroy(s_ArcMaterial);
		delete s_ArcVertexData;
		delete s_Arcs;
		delete s_ArcVertexBuffer;
	}

	void Gizmos::Begin()
	{
		s_LineCount = 0;
		s_ArcCount = 0;
		s_LineVertexDataPtr = s_LineVertexData;
		s_ArcVertexDataPtr = s_ArcVertexData;
	}

	void Gizmos::End()
	{
		FlushLines();
		FlushArcs();
	}

	void Gizmos::SetColor(const Color& color)
	{
		s_CurrentColor = color;
	}

	void Gizmos::SetMatrix(const Matrix& matrix)
	{
		PerDrawConstantBuffer::BindData(matrix);
	}

	void Gizmos::DrawLine(const Vector3& start, const Vector3& end)
	{
		if (s_LineCount >= MAX_LINES)
			FlushLines();

		s_Lines[s_LineCount] = { start, end, s_CurrentColor };
		++s_LineCount;
	}

	void Gizmos::DrawArc(const Vector3& center, const Vector3& normal, const Vector3& from, const float& angle, const float& radius)
	{
		if (s_ArcCount + 1 >= MAX_LINES)
			FlushArcs();

		s_Arcs[s_ArcCount++] = { center, normal, from, radius, angle, s_CurrentColor };
	}

	void Gizmos::DrawBox(const Vector3& center, const Vector3& size)
	{
		if (s_LineCount + 12 >= MAX_LINES)
			FlushLines();

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

		s_Lines[s_LineCount++] = { center + c0, center + c1, s_CurrentColor };
		s_Lines[s_LineCount++] = { center + c1, center + c2, s_CurrentColor };
		s_Lines[s_LineCount++] = { center + c2, center + c3, s_CurrentColor };
		s_Lines[s_LineCount++] = { center + c3, center + c0, s_CurrentColor };

		s_Lines[s_LineCount++] = { center + c4, center + c5, s_CurrentColor };
		s_Lines[s_LineCount++] = { center + c5, center + c6, s_CurrentColor };
		s_Lines[s_LineCount++] = { center + c6, center + c7, s_CurrentColor };
		s_Lines[s_LineCount++] = { center + c7, center + c4, s_CurrentColor };

		s_Lines[s_LineCount++] = { center + c0, center + c4, s_CurrentColor };
		s_Lines[s_LineCount++] = { center + c1, center + c5, s_CurrentColor };
		s_Lines[s_LineCount++] = { center + c2, center + c6, s_CurrentColor };
		s_Lines[s_LineCount++] = { center + c3, center + c7, s_CurrentColor };
	}

	void Gizmos::DrawCapsule(const Vector3& center, const float& height, const float& radius)
	{
		float halfHeight = height / 2;
		float topHeight = center.y + halfHeight;
		float bottomHeight = center.y - halfHeight;

		DrawLine(Vector3(center.x - radius, topHeight, center.z), Vector3(center.x - radius, bottomHeight, center.z));
		DrawLine(Vector3(center.x + radius, topHeight, center.z), Vector3(center.x + radius, bottomHeight, center.z));
		DrawLine(Vector3(center.x, topHeight, center.z - radius), Vector3(center.x, bottomHeight, center.z - radius));
		DrawLine(Vector3(center.x, topHeight, center.z + radius), Vector3(center.x, bottomHeight, center.z + radius));

		DrawArc(Vector3(center.x, topHeight, center.z), -Vector3::UnitX, Vector3::UnitZ, 180, radius);
		DrawArc(Vector3(center.x, topHeight, center.z), Vector3::UnitZ, Vector3::UnitX, 180, radius);
		DrawArc(Vector3(center.x, bottomHeight, center.z), Vector3::UnitX, Vector3::UnitZ, 180, radius);
		DrawArc(Vector3(center.x, bottomHeight, center.z), -Vector3::UnitZ, Vector3::UnitX, 180, radius);

		DrawDisc(Vector3(center.x, topHeight, center.z), Vector3::UnitY, radius);
		DrawDisc(Vector3(center.x, bottomHeight, center.z), Vector3::UnitY, radius);
	}

	void Gizmos::DrawSphere(const Vector3& center, const float& radius)
	{
		if (s_ArcCount + 3 >= MAX_LINES)
			FlushArcs();

		s_Arcs[s_ArcCount++] = { center, Vector3(0, 1, 0), Vector3::UnitX, radius, 360, s_CurrentColor };
		s_Arcs[s_ArcCount++] = { center, Vector3(0, 0, 1), Vector3::UnitX, radius, 360, s_CurrentColor };
		s_Arcs[s_ArcCount++] = { center, Vector3(1, 0, 0), Vector3::UnitZ, radius, 360, s_CurrentColor };
	}

	void Gizmos::DrawDisc(const Vector3& center, const Vector3& normal, const float& radius)
	{
		if (s_ArcCount + 1 >= MAX_LINES)
			FlushArcs();

		Vector3 tangent = normal.Cross(Vector3::Up);
		if (tangent.LengthSquared() < 0.001f)
		{
			tangent = normal.Cross(Vector3::Right);
		}
		s_Arcs[s_ArcCount++] = { center, normal, tangent, radius, 360, s_CurrentColor };
	}

	void Gizmos::DrawFrustum(const Frustum& frustum)
	{
		if (s_LineCount + 12 >= MAX_LINES)
			FlushLines();

		Vector3 corners[8];
		frustum.GetCorners(corners);

		s_Lines[s_LineCount++] = { corners[0], corners[1], s_CurrentColor };
		s_Lines[s_LineCount++] = { corners[1], corners[2], s_CurrentColor };
		s_Lines[s_LineCount++] = { corners[2], corners[3], s_CurrentColor };
		s_Lines[s_LineCount++] = { corners[3], corners[0], s_CurrentColor };

		s_Lines[s_LineCount++] = { corners[4], corners[5], s_CurrentColor };
		s_Lines[s_LineCount++] = { corners[5], corners[6], s_CurrentColor };
		s_Lines[s_LineCount++] = { corners[6], corners[7], s_CurrentColor };
		s_Lines[s_LineCount++] = { corners[7], corners[4], s_CurrentColor };

		s_Lines[s_LineCount++] = { corners[0], corners[4], s_CurrentColor };
		s_Lines[s_LineCount++] = { corners[1], corners[5], s_CurrentColor };
		s_Lines[s_LineCount++] = { corners[2], corners[6], s_CurrentColor };
		s_Lines[s_LineCount++] = { corners[3], corners[7], s_CurrentColor };
	}

	void Gizmos::FlushLines()
	{
		if (s_LineCount == 0)
		{
			return;
		}

		for (uint32_t i = 0; i < s_LineCount; i++)
		{
			Line line = s_Lines[i];
			Vector3 start = line.start;
			Vector3 end = line.end;
			Color color = line.color;

			s_LineVertexDataPtr[0] = start.x;
			s_LineVertexDataPtr[1] = start.y;
			s_LineVertexDataPtr[2] = start.z;

			s_LineVertexDataPtr[3] = color.x;
			s_LineVertexDataPtr[4] = color.y;
			s_LineVertexDataPtr[5] = color.z;
			s_LineVertexDataPtr[6] = color.w;

			s_LineVertexDataPtr[7] = line.end.x;
			s_LineVertexDataPtr[8] = line.end.y;
			s_LineVertexDataPtr[9] = line.end.z;

			s_LineVertexDataPtr[10] = color.x;
			s_LineVertexDataPtr[11] = color.y;
			s_LineVertexDataPtr[12] = color.z;
			s_LineVertexDataPtr[13] = color.w;

			s_LineVertexDataPtr += 14;
		}

		s_LineVertexBuffer->SetData(s_LineVertexData, s_LineCount * 2 * s_LineVertexBuffer->GetElementSize());

		GfxDevice::Draw(GfxDrawingOperation(s_LineVertexBuffer, nullptr, s_LineMaterial, &s_LineLayout, 0, 0, s_LineCount * 2, Topology::LineList, 0));
		GfxDevice::Draw(GfxDrawingOperation(s_LineVertexBuffer, nullptr, s_LineMaterial, &s_LineLayout, 0, 0, s_LineCount * 2, Topology::LineList, 1));
		
		s_LineCount = 0;
		s_LineVertexDataPtr = s_LineVertexData;
	}

	void Gizmos::FlushArcs()
	{
		if (s_ArcCount == 0)
		{
			return;
		}

		for (uint32_t i = 0; i < s_ArcCount; i++)
		{
			Arc arc = s_Arcs[i];
			Vector3 center = arc.center;
			Vector3 normal = arc.normal;
			Vector3 from = arc.from;
			float radius = arc.radius;
			float angle = arc.angle;
			Color color = arc.color;

			s_ArcVertexDataPtr[0] = center.x;
			s_ArcVertexDataPtr[1] = center.y;
			s_ArcVertexDataPtr[2] = center.z;

			s_ArcVertexDataPtr[3] = normal.x;
			s_ArcVertexDataPtr[4] = normal.y;
			s_ArcVertexDataPtr[5] = normal.z;
			s_ArcVertexDataPtr[6] = radius;

			s_ArcVertexDataPtr[7] = from.x;
			s_ArcVertexDataPtr[8] = from.y;
			s_ArcVertexDataPtr[9] = from.z;
			s_ArcVertexDataPtr[10] = angle;

			s_ArcVertexDataPtr[11] = color.x;
			s_ArcVertexDataPtr[12] = color.y;
			s_ArcVertexDataPtr[13] = color.z;
			s_ArcVertexDataPtr[14] = color.w;

			s_ArcVertexDataPtr += 15;
		}

		s_ArcVertexBuffer->SetData(s_ArcVertexData, s_ArcCount * s_ArcVertexBuffer->GetElementSize());

		GfxDevice::Draw(GfxDrawingOperation(s_ArcVertexBuffer, nullptr, s_ArcMaterial, &s_ArcLayout, 0, 0, s_ArcCount, Topology::PointList, 0));
		GfxDevice::Draw(GfxDrawingOperation(s_ArcVertexBuffer, nullptr, s_ArcMaterial, &s_ArcLayout, 0, 1, s_ArcCount, Topology::PointList, 1));
	}
}
