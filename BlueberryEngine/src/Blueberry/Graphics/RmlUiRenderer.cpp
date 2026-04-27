#include "Blueberry\Graphics\RmlUiRenderer.h"

#include "Blueberry\Core\Base.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\VertexLayout.h"
#include "Blueberry\Graphics\Buffers\PerDrawDataConstantBuffer.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Font.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "RmlUiInterfaces.h"

#include <RmlUi\Core.h>
#include <RmlUi\Core\RenderInterface.h>

namespace Blueberry
{
	Material* RmlUiRenderer::s_Material = nullptr;
	VertexLayout* RmlUiRenderer::s_VertexLayout = nullptr;
	RmlUiRenderData* RmlUiRenderer::s_CurrentData = nullptr;
	static Font* s_Font = nullptr;

	static RmlUiSystemInterface* s_SystemInterface = nullptr;
	static RmlUiRenderInterface* s_RenderInterface = nullptr;

	void RmlUiRenderer::Initialize()
	{
		s_SystemInterface = new RmlUiSystemInterface();
		s_RenderInterface = new RmlUiRenderInterface();
		Rml::SetSystemInterface(s_SystemInterface);
		Rml::SetRenderInterface(s_RenderInterface);

		if (!Rml::Initialise())
		{
			BB_ERROR("Failed to initialize RmlUi");
			return;
		}

		s_Material = Material::Create(static_cast<Shader*>(AssetLoader::Load("assets/shaders/Ui.shader")));

		s_VertexLayout = new VertexLayout();
		s_VertexLayout->Append(VertexAttribute::Position, sizeof(Vector2));
		s_VertexLayout->Append(VertexAttribute::Texcoord0, sizeof(Vector2));
		s_VertexLayout->Append(VertexAttribute::Color, sizeof(Color));
		s_VertexLayout->Apply();

		s_Font = static_cast<Font*>(AssetLoader::Load("assets/fonts/segoeui/segoeui.ttf"));

		if (!Rml::LoadFontFace(Rml::Span<const Rml::byte>(s_Font->GetData().data(), s_Font->GetData().size()), "default", Rml::Style::FontStyle::Normal, Rml::Style::FontWeight::Normal))
		{
			BB_ERROR("Failed to load font.");
		}
	}

	void RmlUiRenderer::Shutdown()
	{
		Rml::Shutdown();
		delete s_SystemInterface;
		delete s_RenderInterface;
	}
}