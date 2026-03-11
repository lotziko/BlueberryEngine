#include "SceneObjectPicker.h"

#include "Editor\Selection.h"
#include "Editor\Inspector\ObjectEditor.h"
#include "Editor\Inspector\ObjectEditorDB.h"
#include "Blueberry\Core\Screen.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Scene\Components\SkinnedMeshRenderer.h"
#include "Blueberry\Scene\Components\SpriteRenderer.h"
#include "Blueberry\Graphics\Renderer2D.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Graphics\Buffers\PerCameraDataConstantBuffer.h"
#include "Blueberry\Graphics\Buffers\PerDrawDataConstantBuffer.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\Skinning.h"
#include "PerObjectDataConstantBuffer.h"

namespace Blueberry
{
	Color ConvertIndexToColor(uint16_t index)
	{
		Color result;
		float packedIndex[2] = { (float)((index >> 8) & 255) / 255.0f, (float)(index & 255) / 255.0f };
		memcpy(&result, packedIndex, sizeof(float) * 2);
		return result;
	}

	SceneObjectPicker::SceneObjectPicker()
	{
		TextureProperties textureProperties = {};
		textureProperties.width = Screen::GetWidth();
		textureProperties.height = Screen::GetHeight();
		textureProperties.format = TextureFormat::R8G8B8A8_UNorm;
		textureProperties.usageFlags = TextureUsageFlags::RenderTarget | TextureUsageFlags::CPUReadable;
		GfxDevice::CreateTexture(textureProperties, m_SceneRenderTarget);

		textureProperties.usageFlags = TextureUsageFlags::RenderTarget;
		textureProperties.format = TextureFormat::D24_UNorm;
		GfxDevice::CreateTexture(textureProperties, m_SceneDepthStencil);

		m_SpriteObjectPickerMaterial = Material::Create(static_cast<Shader*>(AssetLoader::Load("assets/shaders/SpriteObjectPicker.shader")));
		m_MeshObjectPickerMaterial = Material::Create(static_cast<Shader*>(AssetLoader::Load("assets/shaders/MeshObjectPicker.shader")));
		m_IconObjectPickerMaterial = Material::Create(static_cast<Shader*>(AssetLoader::Load("assets/shaders/IconObjectPicker.shader")));
		m_ObjectPickerOutlineMaterial = Material::Create(static_cast<Shader*>(AssetLoader::Load("assets/shaders/ObjectPickerOutline.shader")));
	}

	SceneObjectPicker::~SceneObjectPicker()
	{
		delete m_SceneRenderTarget;
		delete m_SceneDepthStencil;
		Object::Destroy(m_SpriteObjectPickerMaterial);
		Object::Destroy(m_MeshObjectPickerMaterial);
		Object::Destroy(m_IconObjectPickerMaterial);
		Object::Destroy(m_ObjectPickerOutlineMaterial);
	}

	Object* SceneObjectPicker::Pick(Scene* scene, Camera* camera, const int& positionX, const int& positionY)
	{
		if (scene == nullptr)
		{
			return nullptr;
		}

		PerCameraDataConstantBuffer::BindData(camera, m_SceneRenderTarget);

		Rectangle area = Rectangle(std::min(std::max(positionX, 0), static_cast<int>(camera->GetPixelSize().x)), std::min(std::max(positionY, 0), static_cast<int>(camera->GetPixelSize().y)), 1, 1);
		unsigned char pixel[4];
		Dictionary<int, ObjectId> validObjects;
		uint32_t index = 1;

		GfxDevice::SetRenderTarget(m_SceneRenderTarget, m_SceneDepthStencil);
		GfxDevice::SetViewport(0, 0, static_cast<int>(camera->GetPixelSize().x), static_cast<int>(camera->GetPixelSize().y));
		GfxDevice::ClearColor({ 0, 0, 0, 0 });
		GfxDevice::ClearDepth(1.0f);
		Renderer2D::Begin();
		for (auto& pair : scene->GetIterator<SpriteRenderer>())
		{
			Entity* entity = pair.second->GetEntity();
			if (entity->IsActiveInHierarchy())
			{
				SpriteRenderer* spriteRenderer = static_cast<SpriteRenderer*>(pair.second);
				if (spriteRenderer->GetTexture() != nullptr)
				{
					Renderer2D::Draw(spriteRenderer->GetLocalToWorldMatrix(), spriteRenderer->GetTexture(), m_SpriteObjectPickerMaterial, ConvertIndexToColor(index), spriteRenderer->GetSortingOrder());
					validObjects[index] = entity->GetObjectId();
					++index;
				}
			}
		}
		Renderer2D::End();

		for (auto& pair : scene->GetIterator<MeshRenderer>())
		{
			Entity* entity = pair.second->GetEntity();
			MeshRenderer* meshRenderer = static_cast<MeshRenderer*>(pair.second);
			Mesh* mesh = meshRenderer->GetMesh();
			if (mesh != nullptr)
			{
				PerDrawDataConstantBuffer::BindData(meshRenderer->GetLocalToWorldMatrix());
				PerObjectDataConstantBuffer::BindData(ConvertIndexToColor(index));
				GfxDevice::Draw(GfxDrawingOperation(mesh, m_MeshObjectPickerMaterial));
				validObjects[index] = entity->GetObjectId();
				++index;
			}
		}

		for (auto& pair : scene->GetIterator<SkinnedMeshRenderer>())
		{
			Entity* entity = pair.second->GetEntity();
			SkinnedMeshRenderer* skinnedMeshRenderer = static_cast<SkinnedMeshRenderer*>(pair.second);
			Mesh* mesh = skinnedMeshRenderer->GetMesh();
			if (mesh != nullptr)
			{
				PerDrawDataConstantBuffer::BindData(skinnedMeshRenderer->GetLocalToWorldMatrix());
				PerObjectDataConstantBuffer::BindData(ConvertIndexToColor(index));
				GfxDevice::Draw(GfxDrawingOperation(mesh, Skinning::GetVertexBuffer(skinnedMeshRenderer), m_MeshObjectPickerMaterial));
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
				for (uint32_t i = 0; i < entity->GetComponentCount(); ++i)
				{
					Component* component = entity->GetComponentAt(i);
					ObjectEditor* editor = ObjectEditor::GetDefaultEditor(component);
					Texture* icon;
					if ((icon = editor->GetIcon(component)) != nullptr)
					{
						m_IconObjectPickerMaterial->SetTexture("_BaseMap", icon);
						Vector3 position = entity->GetTransform()->GetPosition();
						Matrix modelMatrix = Matrix::CreateScale(-0.75f, 0.75f, 0.75f) * Matrix::CreateBillboard(position, position + cameraDirection, Vector3(0, 1, 0));
						PerDrawDataConstantBuffer::BindData(modelMatrix);
						PerObjectDataConstantBuffer::BindData(ConvertIndexToColor(index));
						GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), m_IconObjectPickerMaterial));
						validObjects[index] = entity->GetObjectId();
						++index;
						break;
					}
				}
			}
		}

		GfxDevice::SetRenderTarget(nullptr);
		m_SceneRenderTarget->GetData(pixel, area);

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
		static size_t pickingTextureId = TO_HASH("_PickingTexture");

		if (scene == nullptr)
		{
			return;
		}

		PerCameraDataConstantBuffer::BindData(camera, renderTarget);

		GfxDevice::SetRenderTarget(m_SceneRenderTarget);
		GfxDevice::SetViewport(0, 0, static_cast<int>(camera->GetPixelSize().x), static_cast<int>(camera->GetPixelSize().y));
		GfxDevice::ClearColor({ 0, 0, 0, 0 });
		GfxDevice::ClearDepth(1.0f);

		Renderer2D::Begin();
		for (auto& pair : scene->GetIterator<SpriteRenderer>())
		{
			Entity* entity = pair.second->GetEntity();
			if (Selection::IsActiveObject(entity) && entity->IsActiveInHierarchy())
			{
				SpriteRenderer* spriteRenderer = static_cast<SpriteRenderer*>(pair.second);
				if (spriteRenderer->GetTexture() != nullptr)
				{
					Renderer2D::Draw(spriteRenderer->GetLocalToWorldMatrix(), spriteRenderer->GetTexture(), m_SpriteObjectPickerMaterial, ConvertIndexToColor(65535), spriteRenderer->GetSortingOrder());
				}
			}
		}
		Renderer2D::End();

		for (auto& pair : scene->GetIterator<MeshRenderer>())
		{
			Entity* entity = pair.second->GetEntity();
			if (Selection::IsActiveObject(entity) && entity->IsActiveInHierarchy())
			{
				MeshRenderer* meshRenderer = static_cast<MeshRenderer*>(pair.second);
				Mesh* mesh = meshRenderer->GetMesh();
				if (mesh != nullptr)
				{
					PerDrawDataConstantBuffer::BindData(meshRenderer->GetLocalToWorldMatrix());
					PerObjectDataConstantBuffer::BindData(ConvertIndexToColor(10000));
					GfxDevice::Draw(GfxDrawingOperation(mesh, m_MeshObjectPickerMaterial));
				}
			}
		}

		for (auto& pair : scene->GetIterator<SkinnedMeshRenderer>())
		{
			Entity* entity = pair.second->GetEntity();
			if (Selection::IsActiveObject(entity) && entity->IsActiveInHierarchy())
			{
				SkinnedMeshRenderer* skinnedMeshRenderer = static_cast<SkinnedMeshRenderer*>(pair.second);
				if (skinnedMeshRenderer->HasRoot())
				{
					Mesh* mesh = skinnedMeshRenderer->GetMesh();
					if (mesh != nullptr)
					{
						PerDrawDataConstantBuffer::BindData(skinnedMeshRenderer->GetLocalToWorldMatrix());
						PerObjectDataConstantBuffer::BindData(ConvertIndexToColor(10000));
						GfxDevice::Draw(GfxDrawingOperation(mesh, Skinning::GetVertexBuffer(skinnedMeshRenderer), m_MeshObjectPickerMaterial));
					}
				}
			}
		}
		
		GfxDevice::SetRenderTarget(renderTarget);
		GfxDevice::SetViewport(0, 0, m_SceneRenderTarget->GetWidth(), m_SceneRenderTarget->GetHeight());
		GfxDevice::SetGlobalTexture(pickingTextureId, m_SceneRenderTarget);
		GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), m_ObjectPickerOutlineMaterial));
		GfxDevice::SetRenderTarget(nullptr);
	}
}
