#include "bbpch.h"
#include "Gizmos.h"

#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\VertexLayout.h"

namespace Blueberry
{
	static const UINT MAX_LINES = 1024;
	static const UINT MAX_VERTICES = MAX_LINES * 2;
	static const UINT MAX_INDICES = MAX_LINES * 2;

	bool Gizmos::Initialize()
	{
		Shader* shader = (Shader*)AssetLoader::Load("assets/Color.shader");
		if (shader == nullptr)
		{
			BB_ERROR("Failed to load gizmo line shader.")
			return false;
		}
		s_LineMaterial = Material::Create(shader);

		VertexLayout layout = VertexLayout{}
			.Append(VertexLayout::Position3D)
			.Append(VertexLayout::Float4Color);

		int size = layout.GetSize();
		s_VertexData = new float[MAX_VERTICES * size / sizeof(float)];

		if (!GfxDevice::CreateVertexBuffer(layout, MAX_VERTICES, s_VertexBuffer))
		{
			return false;
		}

		UINT* indexData = new UINT[MAX_INDICES];
		for (UINT i = 0; i < MAX_INDICES; i += 2)
		{
			indexData[i + 0] = i + 0;
			indexData[i + 1] = i + 1;
		}

		if (!GfxDevice::CreateIndexBuffer(MAX_INDICES, s_IndexBuffer))
		{
			return false;
		}
		s_IndexBuffer->SetData(indexData, MAX_INDICES);
		delete[] indexData;

		s_Lines = new Line[MAX_LINES];

		return true;
	}

	void Gizmos::Shutdown()
	{
		Material::Destroy(s_LineMaterial);
		delete s_VertexData;
		delete s_Lines;
		delete s_VertexBuffer;
		delete s_IndexBuffer;
	}

	void Gizmos::Begin()
	{
		s_LineCount = 0;
		s_VertexDataPtr = s_VertexData;
	}

	void Gizmos::End()
	{
		Flush();
	}

	void Gizmos::SetColor(const Color& color)
	{
		s_CurrentColor = color;
	}

	void Gizmos::DrawLine(const Vector3& start, const Vector3& end)
	{
		if (s_LineCount >= MAX_LINES)
			Flush();

		s_Lines[s_LineCount] = { start, end, s_CurrentColor };
		++s_LineCount;
	}

	void Gizmos::DrawBox(const Vector3& center, const Vector3& size)
	{
		if (s_LineCount + 12 >= MAX_LINES)
			Flush();

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

	void Gizmos::DrawCircle(const Vector3& center, const float& radius)
	{
		int segmentCount = 36;

		if (s_LineCount + segmentCount * 3 >= MAX_LINES)
			Flush();

		for (int i = 0; i < segmentCount; i++)
		{
			float x1 = sin(ToRadians((float)i / segmentCount * 360));
			float y1 = cos(ToRadians((float)i / segmentCount * 360));

			float x2 = sin(ToRadians((float)(i + 1) / segmentCount * 360));
			float y2 = cos(ToRadians((float)(i + 1) / segmentCount * 360));

			s_Lines[s_LineCount++] = { center + Vector3(x1 * radius, 0, y1 * radius), center + Vector3(x2 * radius, 0, y2 * radius), s_CurrentColor };
			s_Lines[s_LineCount++] = { center + Vector3(x1 * radius, y1 * radius, 0), center + Vector3(x2 * radius, y2 * radius, 0), s_CurrentColor };
			s_Lines[s_LineCount++] = { center + Vector3(0, x1 * radius, y1 * radius), center + Vector3(0, x2 * radius, y2 * radius), s_CurrentColor };
		}
	}

	void Gizmos::DrawFrustum(const Frustum& frustum)
	{
		if (s_LineCount + 12 >= MAX_LINES)
			Flush();

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

	void Gizmos::Flush()
	{
		for (UINT i = 0; i < s_LineCount; i++)
		{
			Line line = s_Lines[i];
			Vector3 start = line.start;
			Vector3 end = line.end;
			Color color = line.color;

			s_VertexDataPtr[0] = start.x;
			s_VertexDataPtr[1] = start.y;
			s_VertexDataPtr[2] = start.z;

			s_VertexDataPtr[3] = color.x;
			s_VertexDataPtr[4] = color.y;
			s_VertexDataPtr[5] = color.z;
			s_VertexDataPtr[6] = color.w;

			s_VertexDataPtr[7] = line.end.x;
			s_VertexDataPtr[8] = line.end.y;
			s_VertexDataPtr[9] = line.end.z;

			s_VertexDataPtr[10] = color.x;
			s_VertexDataPtr[11] = color.y;
			s_VertexDataPtr[12] = color.z;
			s_VertexDataPtr[13] = color.w;

			s_VertexDataPtr += 14;
		}

		s_VertexBuffer->SetData(s_VertexData, s_LineCount * 2);

		GfxDevice::Draw(GfxDrawingOperation(s_VertexBuffer, s_IndexBuffer, s_LineMaterial, s_LineCount * 2, 0, Topology::LineList));
	}
}
