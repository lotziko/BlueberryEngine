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
	Color ConvertIndexToColor(uint16_t index)
	{
		Color result;
		float packedIndex[2] = { (float)((index >> 8) & 255) / 255.0f, (float)(index & 255) / 255.0f };
		memcpy(&result, packedIndex, sizeof(float) * 2);
		return result;
	}

	class PerObjectDataConstantBuffer
	{
	public:
		struct CONSTANTS
		{
			Color objectId;
		};

		static void BindData(Color indexColor)
		{
			static size_t objectDataId = TO_HASH("PerObjectData");

			if (s_ConstantBuffer == nullptr)
			{
				GfxDevice::CreateConstantBuffer(sizeof(CONSTANTS) * 1, s_ConstantBuffer);
			}

			CONSTANTS constants
			{
				indexColor
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
		properties.format = TextureFormat::R8G8B8A8_UINT;
		properties.isReadable = true;
		GfxDevice::CreateTexture(properties, m_PixelRenderTarget);

		m_SpriteObjectPickerMaterial = Material::Create((Shader*)AssetLoader::Load("assets/SpriteObjectPicker.shader"));
		m_MeshObjectPickerMaterial = Material::Create((Shader*)AssetLoader::Load("assets/MeshObjectPicker.shader"));
		m_ObjectPickerOutlineMaterial = Material::Create((Shader*)AssetLoader::Load("assets/ObjectPickerOutline.shader"));
	}

	SceneObjectPicker::~SceneObjectPicker()
	{
		delete m_SceneRenderTarget;
		delete m_PixelRenderTarget;
	}

	Object* SceneObjectPicker::Pick(Scene* scene, BaseCamera& camera, const int& positionX, const int& positionY)
	{
		if (scene == nullptr)
		{
			return nullptr;
		}

		Rectangle area = Rectangle(Min(Max(positionX, 0), camera.GetPixelSize().x), Min(Max(positionY, 0), camera.GetPixelSize().y), 1, 1);
		unsigned char pixel[4];
		std::unordered_map<int, ObjectId> validObjects;
		uint32_t index = 1;

		GfxDevice::SetRenderTarget(m_SceneRenderTarget, m_SceneDepthStencil);
		GfxDevice::SetViewport(0, 0, camera.GetPixelSize().x, camera.GetPixelSize().y);
		GfxDevice::ClearColor({ 0, 0, 0, 0 });
		GfxDevice::ClearDepth(1.0f);
		Renderer2D::Begin();
		for (auto component : scene->GetIterator<SpriteRenderer>())
		{
			auto spriteRenderer = static_cast<SpriteRenderer*>(component.second);
			if (spriteRenderer->GetTexture() != nullptr)
			{
				Renderer2D::Draw(spriteRenderer->GetEntity()->GetTransform()->GetLocalToWorldMatrix(), spriteRenderer->GetTexture(), m_SpriteObjectPickerMaterial, ConvertIndexToColor(index), spriteRenderer->GetSortingOrder());
				validObjects[index] = spriteRenderer->GetEntity()->GetObjectId();
				++index;
			}
		}
		Renderer2D::End();

		for (auto component : scene->GetIterator<MeshRenderer>())
		{
			auto meshRenderer = static_cast<MeshRenderer*>(component.second);
			Mesh* mesh = meshRenderer->GetMesh();
			if (mesh != nullptr)
			{
				PerDrawConstantBuffer::BindData(meshRenderer->GetEntity()->GetTransform()->GetLocalToWorldMatrix());
				PerObjectDataConstantBuffer::BindData(ConvertIndexToColor(index));
				GfxDevice::Draw(GfxDrawingOperation(mesh, m_MeshObjectPickerMaterial));
				validObjects[index] = meshRenderer->GetEntity()->GetObjectId();
				++index;
			}
		}

		GfxDevice::SetRenderTarget(nullptr);
		GfxDevice::Copy(m_SceneRenderTarget, m_PixelRenderTarget, area);
		m_PixelRenderTarget->GetData(pixel);

		if (pixel[0] > 0 || pixel[1] > 0)
		{
			uint16_t index = (pixel[0] << 8) + (pixel[1]);
			return ObjectDB::GetObject(validObjects[index]);
		}
		else
		{
			return nullptr;
		}
	}

	void SceneObjectPicker::DrawOutline(Scene* scene, BaseCamera& camera, GfxTexture* renderTarget)
	{
		static size_t pickingTextureId = TO_HASH("_PickingTexture");

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
			if (Selection::IsActiveObject(component.second->GetEntity()))
			{
				auto spriteRenderer = static_cast<SpriteRenderer*>(component.second);
				if (spriteRenderer->GetTexture() != nullptr)
				{
					Renderer2D::Draw(spriteRenderer->GetEntity()->GetTransform()->GetLocalToWorldMatrix(), spriteRenderer->GetTexture(), m_SpriteObjectPickerMaterial, ConvertIndexToColor(65535), spriteRenderer->GetSortingOrder());
				}
			}
		}
		Renderer2D::End();

		for (auto component : scene->GetIterator<MeshRenderer>())
		{
			if (Selection::IsActiveObject(component.second->GetEntity()))
			{
				auto meshRenderer = static_cast<MeshRenderer*>(component.second);
				Mesh* mesh = meshRenderer->GetMesh();
				if (mesh != nullptr)
				{
					PerDrawConstantBuffer::BindData(meshRenderer->GetEntity()->GetTransform()->GetLocalToWorldMatrix());
					PerObjectDataConstantBuffer::BindData(ConvertIndexToColor(10000));
					GfxDevice::Draw(GfxDrawingOperation(mesh, m_MeshObjectPickerMaterial));
				}
			}
		}
		
		GfxDevice::SetRenderTarget(renderTarget);
		GfxDevice::SetViewport(0, 0, 1920, 1080);
		GfxDevice::SetGlobalTexture(pickingTextureId, m_SceneRenderTarget);
		GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), m_ObjectPickerOutlineMaterial));
		GfxDevice::SetRenderTarget(nullptr);
	}
}
