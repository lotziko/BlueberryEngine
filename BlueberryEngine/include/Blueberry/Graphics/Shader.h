#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\DataList.h"
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

		const uint32_t GetMask(const size_t& id);

	private:
		Dictionary<size_t, uint32_t> m_KeywordMask = {};
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

		const PropertyType& GetType() const;
		void SetType(const PropertyType& type);

		const String& GetDefaultTextureName() const;
		void SetDefaultTextureName(const String& name);

		const TextureDimension& GetTextureDimension() const;
		void SetTextureDimension(const TextureDimension& dimension);

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

		const CullMode& GetCullMode() const;
		void SetCullMode(const CullMode& cullMode);

		const BlendMode& GetBlendSrcColor() const;
		const BlendMode& GetBlendSrcAlpha() const;
		void SetBlendSrc(const BlendMode& blendSrc);
		void SetBlendSrc(const BlendMode& blendSrcColor, const BlendMode& blendSrcAlpha);

		const BlendMode& GetBlendDstColor() const;
		const BlendMode& GetBlendDstAlpha() const;
		void SetBlendDst(const BlendMode& blendDst);
		void SetBlendDst(const BlendMode& blendDstColor, const BlendMode& blendDstAlpha);

		const ZTest& GetZTest() const;
		void SetZTest(const ZTest& zTest);

		const ZWrite& GetZWrite() const;
		void SetZWrite(const ZWrite& zWrite);

		const List<String>& GetVertexKeywords() const;
		void SetVertexKeywords(const List<String>& keywords);

		const List<String>& GetFragmentKeywords() const;
		void SetFragmentKeywords(const List<String>& keywords);

		const uint32_t& GetVertexOffset() const;
		void SetVertexOffset(const uint32_t& offset);

		const uint32_t& GetGeometryOffset() const;
		void SetGeometryOffset(const uint32_t& offset);

		const uint32_t& GetFragmentOffset() const;
		void SetFragmentOffset(const uint32_t& offset);

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

		const PassData& GetPass(const uint32_t& index) const;
		const size_t GetPassCount() const;
		void SetPasses(const DataList<PassData>& passes);

		const DataList<PropertyData>& GetProperties() const;
		void SetProperties(const DataList<PropertyData>& properties);

	private:
		DataList<PassData> m_Passes;
		DataList<PropertyData> m_Properties;
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
		virtual ~Shader() = default;

		const ShaderData& GetData() const;

		void Initialize(const VariantsData& variantsData);
		void Initialize(const VariantsData& variantsData, const ShaderData& data);

		static Shader* Create(const VariantsData& variantsData, const ShaderData& shaderData, Shader* existingShader = nullptr);

		static void SetKeyword(const size_t& id, const bool& enabled);
		static const uint32_t& GetActiveKeywordsMask();

	private:
		const Shader::ShaderVariant GetVariant(const uint32_t& vertexKeywordFlags, const uint32_t& fragmentKeywordFlags, const uint8_t& passIndex);
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

		static HashSet<size_t> s_ActiveKeywords;
		static uint32_t s_ActiveKeywordsMask;
		static KeywordDB s_GlobalKeywords;

		friend struct GfxDrawingOperation;
		friend class Material;
		friend class GfxRenderStateCache;
	};
}