#include "bbpch.h"
#include "SceneObjectPicker.h"

#include "Editor\Selection.h"
#include "Editor\Inspector\ObjectInspector.h"
#include "Editor\Inspector\ObjectInspectorDB.h"
#include "Blueberry\Core\Screen.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Graphics\Renderer2D.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Graphics\PerCameraDataConstantBuffer.h"
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
			static std::size_t objectDataId = TO_HASH("PerObjectData");

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

	SceneObjectPicker::SceneObjectPicker()
	{
		TextureProperties properties = {};
		properties.width = Screen::GetWidth();
		properties.height = Screen::GetHeight();
		properties.isRenderTarget = true;
		properties.isReadable = true;
		properties.format = TextureFormat::R8G8B8A8_UNorm;
		GfxDevice::CreateTexture(properties, m_SceneRenderTarget);

		properties.isReadable = false;
		properties.format = TextureFormat::D24_UNorm;
		GfxDevice::CreateTexture(properties, m_SceneDepthStencil);

		m_SpriteObjectPickerMaterial = Material::Create((Shader*)AssetLoader::Load("assets/shaders/SpriteObjectPicker.shader"));
		m_MeshObjectPickerMaterial = Material::Create((Shader*)AssetLoader::Load("assets/shaders/MeshObjectPicker.shader"));
		m_ObjectPickerOutlineMaterial = Material::Create((Shader*)AssetLoader::Load("assets/shaders/ObjectPickerOutline.shader"));
	}

	SceneObjectPicker::~SceneObjectPicker()
	{
		delete m_SceneRenderTarget;
		delete m_SceneDepthStencil;
	}

	Object* SceneObjectPicker::Pick(Scene* scene, Camera* camera, const int& positionX, const int& positionY)
	{
		if (scene == nullptr)
		{
			return nullptr;
		}

		PerCameraDataConstantBuffer::BindData(camera);

		Rectangle area = Rectangle(Min(Max(positionX, 0), camera->GetPixelSize().x), Min(Max(positionY, 0), camera->GetPixelSize().y), 1, 1);
		unsigned char pixel[4];
		std::unordered_map<int, ObjectId> validObjects;
		uint32_t index = 1;

		GfxDevice::SetRenderTarget(m_SceneRenderTarget, m_SceneDepthStencil);
		GfxDevice::SetViewport(0, 0, camera->GetPixelSize().x, camera->GetPixelSize().y);
		GfxDevice::ClearColor({ 0, 0, 0, 0 });
		GfxDevice::ClearDepth(1.0f);
		Renderer2D::Begin();
		for (auto& pair : scene->GetIterator<SpriteRenderer>())
		{
			Entity* entity = pair.second->GetEntity();
			if (entity->IsActiveInHierarchy())
			{
				auto spriteRenderer = static_cast<SpriteRenderer*>(pair.second);
				if (spriteRenderer->GetTexture() != nullptr)
				{
					Renderer2D::Draw(entity->GetTransform()->GetLocalToWorldMatrix(), spriteRenderer->GetTexture(), m_SpriteObjectPickerMaterial, ConvertIndexToColor(index), spriteRenderer->GetSortingOrder());
					validObjects[index] = entity->GetObjectId();
					++index;
				}
			}
		}
		Renderer2D::End();

		for (auto& pair : scene->GetIterator<MeshRenderer>())
		{
			Entity* entity = pair.second->GetEntity();
			auto meshRenderer = static_cast<MeshRenderer*>(pair.second);
			Mesh* mesh = meshRenderer->GetMesh();
			if (mesh != nullptr)
			{
				PerDrawConstantBuffer::BindData(entity->GetTransform()->GetLocalToWorldMatrix());
				PerObjectDataConstantBuffer::BindData(ConvertIndexToColor(index));
				GfxDevice::Draw(GfxDrawingOperation(mesh, m_MeshObjectPickerMaterial));
				validObjects[index] = entity->GetObjectId();
				++index;
			}
		}

		Vector3 cameraDirection = Vector3::Transform(Vector3::Forward, camera->GetTransform()->GetRotation());
		for (auto& pair : scene->GetEntities())
		{
			Entity* entity = pair.second.Get();
			if (entity->IsActiveInHierarchy())
			{
				for (auto& component : entity->GetComponents())
				{
					ObjectInspector* inspector = ObjectInspectorDB::GetInspector(component->GetType());
					if (inspector->GetIconPath(component) != nullptr)
					{
						Vector3 position = entity->GetTransform()->GetPosition();
						Matrix modelMatrix = Matrix::CreateScale(0.75f) * Matrix::CreateBillboard(position, position - cameraDirection, Vector3(0, -1, 0));
						PerDrawConstantBuffer::BindData(modelMatrix);
						PerObjectDataConstantBuffer::BindData(ConvertIndexToColor(index));
						GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), m_MeshObjectPickerMaterial));
						validObjects[index] = entity->GetObjectId();
						++index;
						break;
					}
				}
			}
		}

		GfxDevice::SetRenderTarget(nullptr);
		GfxDevice::Read(m_SceneRenderTarget, pixel, area);

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

	void SceneObjectPicker::DrawOutline(Scene* scene, Camera* camera, GfxTexture* renderTarget)
	{
		static std::size_t pickingTextureId = TO_HASH("_PickingTexture");

		if (scene == nullptr)
		{
			return;
		}

		PerCameraDataConstantBuffer::BindData(camera);

		GfxDevice::SetRenderTarget(m_SceneRenderTarget);
		GfxDevice::SetViewport(0, 0, camera->GetPixelSize().x, camera->GetPixelSize().y);
		GfxDevice::ClearColor({ 0, 0, 0, 0 });
		GfxDevice::ClearDepth(1.0f);

		Renderer2D::Begin();
		for (auto& pair : scene->GetIterator<SpriteRenderer>())
		{
			Entity* entity = pair.second->GetEntity();
			if (Selection::IsActiveObject(entity) && entity->IsActiveInHierarchy())
			{
				auto spriteRenderer = static_cast<SpriteRenderer*>(pair.second);
				if (spriteRenderer->GetTexture() != nullptr)
				{
					Renderer2D::Draw(entity->GetTransform()->GetLocalToWorldMatrix(), spriteRenderer->GetTexture(), m_SpriteObjectPickerMaterial, ConvertIndexToColor(65535), spriteRenderer->GetSortingOrder());
				}
			}
		}
		Renderer2D::End();

		for (auto& pair : scene->GetIterator<MeshRenderer>()) // REMOVE FROM LIST WHEN DISABLING INSTEAD OF ITERATING OVER DISABLED ONES
		{
			Entity* entity = pair.second->GetEntity();
			if (Selection::IsActiveObject(entity) && entity->IsActiveInHierarchy())
			{
				auto meshRenderer = static_cast<MeshRenderer*>(pair.second);
				Mesh* mesh = meshRenderer->GetMesh();
				if (mesh != nullptr)
				{
					PerDrawConstantBuffer::BindData(entity->GetTransform()->GetLocalToWorldMatrix());
					PerObjectDataConstantBuffer::BindData(ConvertIndexToColor(10000));
					GfxDevice::Draw(GfxDrawingOperation(mesh, m_MeshObjectPickerMaterial));
				}
			}
		}
		
		GfxDevice::SetRenderTarget(renderTarget);
		GfxDevice::SetViewport(0, 0, renderTarget->GetWidth(), renderTarget->GetHeight());
		GfxDevice::SetGlobalTexture(pickingTextureId, m_SceneRenderTarget);
		GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), m_ObjectPickerOutlineMaterial));
		GfxDevice::SetRenderTarget(nullptr);
	}
}
