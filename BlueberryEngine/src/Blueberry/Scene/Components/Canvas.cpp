#include "Blueberry\Scene\Components\Canvas.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Core\Screen.h"
#include "Blueberry\Core\Application.h"
#include "Blueberry\Graphics\RmlUiRenderer.h"
#include "Blueberry\Graphics\Buffers\PerCameraDataConstantBuffer.h"
#include "..\..\Graphics\RmlUiInterfaces.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\UI\UIDocument.h"

#include <RmlUi\Core.h>

namespace Blueberry
{
	OBJECT_DEFINITION(Canvas, Component)
	{
		DEFINE_BASE_FIELDS(Canvas, Component)
		DEFINE_FIELD(Canvas, m_Camera, BindingType::ObjectPtr, FieldOptions().SetObjectType(&Camera::Type))
		DEFINE_FIELD(Canvas, m_Document, BindingType::ObjectPtr, FieldOptions().SetObjectType(&UIDocument::Type))
		DEFINE_ITERATOR(Canvas)
		DEFINE_EXECUTE_ALWAYS()
	}

	void Canvas::OnCreate()
	{
		if (m_Context == nullptr)
		{
			Initialize();
		}
	}

	void Canvas::OnDestroy()
	{
		Rml::RemoveContext(Rml::CreateString("%d", GetObjectId()));
		delete m_RenderData;
		if (Application::IsRunning())
		{
			InputEvents::GetMouseMoved().RemoveCallback<Canvas, &Canvas::OnMouseMove>(this);
		}
	}

	void Canvas::Draw()
	{
		if (m_Camera.IsValid())
		{
			RmlUiRenderer::s_CurrentData = m_RenderData;
			Vector2 cameraSize = m_Camera->GetPixelSize();
			Rml::Vector2i contextDimensions = m_Context->GetDimensions();
			Rml::Vector2i cameraDimensions = Rml::Vector2i(static_cast<int>(cameraSize.x), static_cast<int>(cameraSize.y));
			if (cameraDimensions.x != contextDimensions.x || cameraDimensions.y != contextDimensions.y)
			{
				float scale = cameraDimensions.y / 1080.0f;
				m_Context->SetDimensions(cameraDimensions);
				m_Context->SetDensityIndependentPixelRatio(scale);
				m_ProjectionMatrix = Matrix::CreateOrthographicOffCenter(0, cameraDimensions.x, cameraDimensions.y, 0, 0.01f, 10.0f);
			}

			if (m_UpdateCount != m_Document->GetUpdateCount())
			{
				m_Context->GetDocument(0)->Close();
				Rml::ElementDocument* document = m_Context->LoadDocumentFromMemory(m_Document->GetData().c_str());
				if (document == nullptr)
				{
					BB_ERROR("Failed to reload document");
					return;
				}
				else
				{
					document->Show();
				}
			}

			m_Context->Update();
			PerCameraDataConstantBuffer::BindData(m_ProjectionMatrix);
			m_Context->Render();
		}
	}

	Camera* Canvas::GetCamera() const
	{
		return m_Camera.Get();
	}

	void Canvas::SetCamera(Camera* camera)
	{
		m_Camera = camera;
	}

	Rml::Element* Canvas::GetRoot()
	{
		if (!m_Camera.IsValid())
		{
			return nullptr;
		}
		if (m_Context == nullptr)
		{
			Initialize();
		}
		return m_Context->GetRootElement();
	}

	void Canvas::Initialize()
	{
		if (!m_Camera.IsValid())
		{
			BB_ERROR("Canvas has no camera.");
			return;
		}

		if (!m_Document.IsValid())
		{
			BB_ERROR("Canvas has no document.");
			return;
		}

		Rml::Vector2i dimensions = Rml::Vector2i(Screen::GetWidth(), Screen::GetHeight());
		m_Context = Rml::CreateContext(Rml::CreateString("%d", GetObjectId()), dimensions);
		m_RenderData = new RmlUiRenderData();
		m_ProjectionMatrix = Matrix::CreateOrthographicOffCenter(0, static_cast<float>(dimensions.x), static_cast<float>(dimensions.y), 0, 0.01f, 10.0f);
		m_UpdateCount = m_Document->GetUpdateCount();

		Rml::ElementDocument* document = m_Context->LoadDocumentFromMemory(m_Document->GetData().c_str());

		if (document == nullptr)
		{
			BB_ERROR("Failed to create document");
			return;
		}
		else
		{
			document->Show();
		}
		if (Application::IsRunning())
		{
			InputEvents::GetMouseMoved().AddCallback<Canvas, &Canvas::OnMouseMove>(this);
		}
	}

	void Canvas::OnMouseMove(const MouseMoveEventArgs& args)
	{
		Vector2 position = args.GetPosition() - Screen::GetGameViewport().Location();
		m_Context->ProcessMouseMove(static_cast<int>(position.x), static_cast<int>(position.y), 0);
	}
}