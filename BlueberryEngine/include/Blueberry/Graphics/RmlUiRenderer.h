#pragma once

namespace Blueberry
{
	class Material;
	class VertexLayout;
	class RmlUiRenderData;

	class RmlUiRenderer
	{
	public:
		static void Initialize();
		static void Shutdown();

	private:
		static Material* s_Material;
		static VertexLayout* s_VertexLayout;
		static RmlUiRenderData* s_CurrentData;

		friend class RmlUiRenderInterface;
		friend class Canvas;
	};
}