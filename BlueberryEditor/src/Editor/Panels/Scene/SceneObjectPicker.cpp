#include "bbpch.h"
#include "SceneObjectPicker.h"

#include "Editor\Selection.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\BaseCamera.h"
#include "Blueberry\Graphics\Renderer2D.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Graphics\PerDrawDataConstantBuffer.h"
#include "Blueberry\Graphics\StandardMeshes.h"

namespace Blueberry
{
	class PerObjectDataConstantBuffer
	{
	public:
		struct CONSTANTS
		{
			float objectId;
		};

		static void BindData(float index)
		{
			static size_t objectDataId = TO_HASH("PerObjectData");

			if (s_ConstantBuffer == nullptr)
			{
				GfxDevice::CreateConstantBuffer(sizeof(CONSTANTS) * 1, s_ConstantBuffer);
			}

			CONSTANTS constants =
			{
				index
			};

			s_ConstantBuffer->SetData(reinterpret_cast<char*>(&constants), sizeof(constants));
			GfxDevice::SetGlobalConstantBuffer(objectDataId, s_ConstantBuffer);
		}

	private:
		static inline GfxConstantBuffer* s_ConstantBuffer = nullptr;
	};

	SceneObjectPicker::SceneObjectPicker(GfxTexture* depthStencilTexture) : m_SceneDepthStencil(depthStencilTexture)
	{
		TextureProperties properties = {};
		properties.width = 1920;
		properties.height = 1080;
		properties.isRenderTarget = true;
		properties.format = TextureFormat::R8G8B8A8_UNorm;
		GfxDevice::CreateTexture(properties, m_SceneRenderTarget);

		properties.width = 1;
		properties.height = 1;
		properties.isReadable = true;
		GfxDevice::CreateTexture(properties, m_PixelRenderTarget);

		m_SpriteObjectPickerMaterial = Material::Create((Shader*)AssetLoader::Load("assets/SpriteObjectPicker.shader"));
		m_SpriteObjectPickerMaterial->SetSurfaceType(SurfaceType::Transparent);
		m_MeshObjectPickerMaterial = Material::Create((Shader*)AssetLoader::Load("assets/MeshObjectPicker.shader"));
		m_MeshObjectPickerMaterial->SetSurfaceType(SurfaceType::Transparent);
		m_ObjectPickerOutlineMaterial = Material::Create((Shader*)AssetLoader::Load("assets/ObjectPickerOutline.shader"));
		m_ObjectPickerOutlineMaterial->SetSurfaceType(SurfaceType::Transparent);
	}

	SceneObjectPicker::~SceneObjectPicker()
	{
		delete m_SceneRenderTarget;
		delete m_PixelRenderTarget;
	}

	void SceneObjectPicker::Pick(Scene* scene, BaseCamera& camera, const int& positionX, const int& positionY)
	{
		if (scene == nullptr)
		{
			return;
		}

		Rectangle area = Rectangle(Min(Max(positionX, 0), camera.GetPixelSize().x), Min(Max(positionY, 0), camera.GetPixelSize().y), 1, 1);
		char pixel[4];
		std::map<int, ObjectId> validObjects;
		int index = 1;

		GfxDevice::SetRenderTarget(m_SceneRenderTarget, m_SceneDepthStencil);
		GfxDevice::SetViewport(0, 0, camera.GetPixelSize().x, camera.GetPixelSize().y);
		GfxDevice::ClearColor({ 0, 0, 0, 0 });
		GfxDevice::ClearDepth(1.0f);
		Renderer2D::Begin();
		for (auto component : scene->GetIterator<SpriteRenderer>())
		{
			auto spriteRenderer = static_cast<SpriteRenderer*>(component.second);
			if (spriteRenderer->GetTexture() != nullptr && spriteRenderer->GetMaterial() != nullptr)
			{
				Renderer2D::Draw(spriteRenderer->GetEntity()->GetTransform()->GetLocalToWorldMatrix(), spriteRenderer->GetTexture(), m_SpriteObjectPickerMaterial, Color((float)index / 255.0f, 0, 0), spriteRenderer->GetSortingOrder());
				validObjects[index] = spriteRenderer->GetEntity()->GetObjectId();
				++index;
			}
		}
		Renderer2D::End();

		for (auto component : scene->GetIterator<MeshRenderer>())
		{
			auto meshRenderer = static_cast<MeshRenderer*>(component.second);
			Mesh* mesh = meshRenderer->GetMesh();
			if (mesh != nullptr && meshRenderer->GetMaterial() != nullptr)
			{
				PerDrawConstantBuffer::BindData(meshRenderer->GetEntity()->GetTransform()->GetLocalToWorldMatrix());
				PerObjectDataConstantBuffer::BindData((float)index / 255.0f);
				GfxDevice::Draw(GfxDrawingOperation(mesh, m_MeshObjectPickerMaterial));
				validObjects[index] = meshRenderer->GetEntity()->GetObjectId();
				++index;
			}
		}

		GfxDevice::SetRenderTarget(nullptr);
		GfxDevice::Copy(m_SceneRenderTarget, m_PixelRenderTarget, area);
		m_PixelRenderTarget->GetData(pixel);

		if (pixel[0] > 0)
		{
			Selection::SetActiveObject(ObjectDB::GetObject(validObjects[pixel[0]]));
		}
		else
		{
			Selection::SetActiveObject(nullptr);
		}
	}

	void SceneObjectPicker::DrawOutline(Scene* scene, BaseCamera& camera, GfxTexture* renderTarget)
	{
		static size_t baseMapId = TO_HASH("_BaseMap");

		if (scene == nullptr)
		{
			return;
		}

		GfxDevice::SetRenderTarget(m_SceneRenderTarget);
		GfxDevice::SetViewport(0, 0, camera.GetPixelSize().x, camera.GetPixelSize().y);
		GfxDevice::ClearColor({ 0, 0, 0, 0 });
		GfxDevice::ClearDepth(1.0f);

		Renderer2D::Begin();
		for (auto component : scene->GetIterator<SpriteRenderer>())
		{
			if (component.second->GetEntity() == Selection::GetActiveObject())
			{
				auto spriteRenderer = static_cast<SpriteRenderer*>(component.second);
				if (spriteRenderer->GetTexture() != nullptr && spriteRenderer->GetMaterial() != nullptr)
				{
					Renderer2D::Draw(spriteRenderer->GetEntity()->GetTransform()->GetLocalToWorldMatrix(), spriteRenderer->GetTexture(), m_SpriteObjectPickerMaterial, Color(1.0f, 0, 0), spriteRenderer->GetSortingOrder());
				}
			}
		}
		Renderer2D::End();

		for (auto component : scene->GetIterator<MeshRenderer>())
		{
			if (component.second->GetEntity() == Selection::GetActiveObject())
			{
				auto meshRenderer = static_cast<MeshRenderer*>(component.second);
				Mesh* mesh = meshRenderer->GetMesh();
				if (mesh != nullptr && meshRenderer->GetMaterial() != nullptr)
				{
					PerDrawConstantBuffer::BindData(meshRenderer->GetEntity()->GetTransform()->GetLocalToWorldMatrix());
					PerObjectDataConstantBuffer::BindData(1.0f);
					GfxDevice::Draw(GfxDrawingOperation(mesh, m_MeshObjectPickerMaterial));
				}
			}
		}
		
		GfxDevice::SetRenderTarget(renderTarget);
		GfxDevice::SetViewport(0, 0, 1920, 1080);
		GfxDevice::SetGlobalTexture(baseMapId, m_SceneRenderTarget);
		GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), m_ObjectPickerOutlineMaterial));
		GfxDevice::SetRenderTarget(nullptr);
	}
}
