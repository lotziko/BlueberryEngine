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

	void Gizmos::Flush()
	{
		for (int i = 0; i < s_LineCount; i++)
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
