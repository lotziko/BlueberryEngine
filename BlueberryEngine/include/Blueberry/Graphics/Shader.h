#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Graphics\Enums.h"

namespace Blueberry
{
	class GfxShader;
	class GfxVertexShader;
	class GfxGeometryShader;
	class GfxFragmentShader;

	class KeywordDB
	{
	public:
		KeywordDB() = default;
		virtual ~KeywordDB() = default;

		const uint32_t GetMask(size_t id);

	private:
		List<std::pair<size_t, uint32_t>> m_KeywordMask = {};
		uint32_t m_MaxMask = 1;
	};

	class BB_API PropertyData : public Data
	{
		DATA_DECLARATION(PropertyData)

	public:
		enum class PropertyType
		{
			Texture,
			Float,
			Int,
			Vector,
			Color,
		};

	public:
		PropertyData() = default;
		virtual ~PropertyData() = default;

		const String& GetName() const;
		void SetName(const String& name);

		PropertyType GetType() const;
		void SetType(PropertyType type);

		const String& GetDefaultTextureName() const;
		void SetDefaultTextureName(const String& name);

		TextureDimension GetTextureDimension() const;
		void SetTextureDimension(TextureDimension dimension);

	private:
		String m_Name;
		PropertyType m_Type;

		String m_DefaultTextureName;
		TextureDimension m_TextureDimension;
	};

	class BB_API PassData : public Data
	{
		DATA_DECLARATION(PassData)

	public:
		PassData() = default;
		virtual ~PassData() = default;

		CullMode GetCullMode() const;
		void SetCullMode(CullMode cullMode);

		BlendMode GetBlendSrcColor() const;
		BlendMode GetBlendSrcAlpha() const;
		void SetBlendSrc(BlendMode blendSrc);
		void SetBlendSrc(BlendMode blendSrcColor, BlendMode blendSrcAlpha);

		BlendMode GetBlendDstColor() const;
		BlendMode GetBlendDstAlpha() const;
		void SetBlendDst(BlendMode blendDst);
		void SetBlendDst(BlendMode blendDstColor, BlendMode blendDstAlpha);

		ZTest GetZTest() const;
		void SetZTest(ZTest zTest);

		ZWrite GetZWrite() const;
		void SetZWrite(ZWrite zWrite);

		const List<String>& GetVertexKeywords() const;
		void SetVertexKeywords(const List<String>& keywords);

		const List<String>& GetFragmentKeywords() const;
		void SetFragmentKeywords(const List<String>& keywords);

		uint32_t GetVertexOffset() const;
		void SetVertexOffset(uint32_t offset);

		uint32_t GetGeometryOffset() const;
		void SetGeometryOffset(uint32_t offset);

		uint32_t GetFragmentOffset() const;
		void SetFragmentOffset(uint32_t offset);

	private:
		CullMode m_CullMode = CullMode::Front;
		BlendMode m_SrcBlendColor = BlendMode::One;
		BlendMode m_DstBlendColor = BlendMode::Zero;
		BlendMode m_SrcBlendAlpha = BlendMode::One;
		BlendMode m_DstBlendAlpha = BlendMode::Zero;
		ZTest m_ZTest = ZTest::LessEqual;
		ZWrite m_ZWrite = ZWrite::On;
		List<String> m_VertexKeywords;
		List<String> m_FragmentKeywords;
		uint32_t m_VertexOffset;
		uint32_t m_GeometryOffset;
		uint32_t m_FragmentOffset;
	};

	class Texture2D;

	class BB_API ShaderData : public Data
	{
		DATA_DECLARATION(ShaderData)

	public:
		ShaderData() = default;
		virtual ~ShaderData() = default;

		const PassData& GetPass(uint32_t index) const;
		const size_t GetPassCount() const;
		void SetPasses(const List<PassData>& passes);

		const List<PropertyData>& GetProperties() const;
		void SetProperties(const List<PropertyData>& properties);

	private:
		List<PassData> m_Passes;
		List<PropertyData> m_Properties;
	};

	struct VariantsData
	{
		List<uint32_t> vertexShaderIndices;
		List<uint32_t> geometryShaderIndices;
		List<uint32_t> fragmentShaderIndices;

		List<void*> shaders;
	};

	struct ShaderVariant
	{
		GfxVertexShader* vertexShader;
		GfxGeometryShader* geometryShader;
		GfxFragmentShader* fragmentShader;
	};

	class BB_API Shader : public Object
	{
		OBJECT_DECLARATION(Shader)

	public:

		Shader() = default;
		virtual ~Shader();

		const ShaderData& GetData() const;

		void Initialize(const VariantsData& variantsData);
		void Initialize(const VariantsData& variantsData, const ShaderData& data);

		static Shader* Create(const VariantsData& variantsData, const ShaderData& shaderData);

		static void SetKeyword(size_t id, bool enabled);
		static uint32_t GetActiveKeywordsMask();

	private:
		const Shader::ShaderVariant GetVariant(uint32_t vertexKeywordFlags, uint32_t fragmentKeywordFlags, uint8_t passIndex);
		void IncrementUpdateCount();

	private:
		ShaderData m_Data;

		List<GfxVertexShader*> m_VertexShaders;
		List<GfxGeometryShader*> m_GeometryShaders;
		List<GfxFragmentShader*> m_FragmentShaders;
		List<std::tuple<uint32_t, uint32_t, uint32_t>> m_PassesOffsets;
		KeywordDB m_LocalKeywords = {};
		uint32_t m_UpdateCount = 0;
		HashSet<ObjectId> m_Dependencies;

		static List<size_t> s_ActiveKeywords;
		static uint32_t s_ActiveKeywordsMask;
		static KeywordDB s_GlobalKeywords;

		friend struct GfxDrawingOperation;
		friend class Material;
		friend class GfxRenderStateCache;
	};
}