#pragma once

#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Scene\Components\Component.h"
#include "Blueberry\Events\InputEvents.h"

namespace Rml
{
	class Context;
	class Element;
}

namespace Blueberry
{
	class Camera;
	class UIDocument;
	class RmlUiRenderData;

	class BB_API Canvas : public Component
	{
		OBJECT_DECLARATION(Canvas)

	public:
		Canvas() = default;
		virtual ~Canvas() = default;

		virtual void OnCreate();
		virtual void OnDestroy();

		void Draw();

		Camera* GetCamera() const;
		void SetCamera(Camera* camera);

		Rml::Element* GetRoot();

	private:
		void Initialize();
		void OnMouseMove(const MouseMoveEventArgs& args);

	private:
		ObjectPtr<Camera> m_Camera;
		ObjectPtr<UIDocument> m_Document;

		Rml::Context* m_Context = nullptr;
		RmlUiRenderData* m_RenderData = nullptr;
		Matrix m_ProjectionMatrix;
		size_t m_UpdateCount = 0;
	};
}