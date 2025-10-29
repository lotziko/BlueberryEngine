Shader
{
	HLSLINCLUDE
	#include "Core.hlsl"

	struct Attributes
	{
		float3 positionOS : POSITION;
		float2 texcoord : TEXCOORD0;
		VERTEX_INPUT_INSTANCE_ID
	};

	struct Varyings
	{
		float4 positionCS : SV_POSITION;
		float2 texcoord : TEXCOORD0;
		VERTEX_OUTPUT_VIEW_INDEX
	};

	cbuffer BloomData
	{
		float2 _TexelSize;
		float2 _Dummy;
	};

	Varyings BloomVertex(Attributes input)
	{
		Varyings output;
		SETUP_INSTANCE_ID(input);
		SETUP_OUTPUT_VIEW_INDEX(output);

		output.positionCS = float4(input.positionOS, 1.0f);
		output.texcoord = input.texcoord;
		return output;
	}

	float Luminance(float3 rgb)
	{
		return dot(rgb, float3(0.2126729f, 0.7151522f, 0.0721750f));
	}

	TEXTURE2D_X(_SourceTexture);	SAMPLER(_SourceTexture_Sampler);

	float4 FragBlur(float2 uv, bool isHorizontal, bool useLuminance)
	{
		float3 c0 = SAMPLE_TEXTURE2D_X(_SourceTexture, _SourceTexture_Sampler, uv + (isHorizontal ? float2(-2.196197, 0) : float2(0, -2.196197)) * _TexelSize).rgb;
		float3 c1 = SAMPLE_TEXTURE2D_X(_SourceTexture, _SourceTexture_Sampler, uv + (isHorizontal ? float2(-0.430063, 0) : float2(0, -0.430063)) * _TexelSize).rgb;
		float3 c2 = SAMPLE_TEXTURE2D_X(_SourceTexture, _SourceTexture_Sampler, uv + (isHorizontal ? float2(1, 0) : float2(0, 1)) * _TexelSize).rgb;
		float3 c3 = SAMPLE_TEXTURE2D_X(_SourceTexture, _SourceTexture_Sampler, uv + (isHorizontal ? float2(2.196197, 0) : float2(0, 2.196197)) * _TexelSize).rgb;

		if (useLuminance)
		{
			c0 /= 1 + Luminance(c0);
			c1 /= 1 + Luminance(c1);
			c2 /= 1 + Luminance(c2);
			c3 /= 1 + Luminance(c3);
		}
		float3 col = c0 * 0.121597 + c1 * 0.529213 + c2 * 0.227595 + c3 * 0.121597;
		if (useLuminance)
		{
			col /= 1 - Luminance(col);
		}
		return float4(col, 1.0);
	}
	HLSLEND

	Pass
	{
		Blend One Zero
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex BloomVertex
		#pragma fragment BloomFragment

		#pragma keyword_global_vertex MULTIVIEW
		#pragma keyword_global_fragment MULTIVIEW

		float4 BloomFragment(Varyings input) : SV_TARGET
		{
			SETUP_INPUT_VIEW_INDEX(input);
			return FragBlur(input.texcoord, true, true);
		}
		HLSLEND
	}
	Pass
	{
		Blend One Zero
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex BloomVertex
		#pragma fragment BloomFragment

		#pragma keyword_global_vertex MULTIVIEW
		#pragma keyword_global_fragment MULTIVIEW

		float4 BloomFragment(Varyings input) : SV_TARGET
		{
			SETUP_INPUT_VIEW_INDEX(input);
			return FragBlur(input.texcoord, false, true);
		}
		HLSLEND
	}
	Pass
	{
		Blend One Zero
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex BloomVertex
		#pragma fragment BloomFragment

		#pragma keyword_global_vertex MULTIVIEW
		#pragma keyword_global_fragment MULTIVIEW

		float4 BloomFragment(Varyings input) : SV_TARGET
		{
			SETUP_INPUT_VIEW_INDEX(input);
			return FragBlur(input.texcoord, true, false);
		}
		HLSLEND
	}
	Pass
	{
		Blend One Zero
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex BloomVertex
		#pragma fragment BloomFragment

		#pragma keyword_global_vertex MULTIVIEW
		#pragma keyword_global_fragment MULTIVIEW

		float4 BloomFragment(Varyings input) : SV_TARGET
		{
			SETUP_INPUT_VIEW_INDEX(input);
			return FragBlur(input.texcoord, false, false);
		}
		HLSLEND
	}
	Pass
	{
		Blend One Zero
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex BloomVertex
		#pragma fragment BloomFragment

		#pragma keyword_global_vertex MULTIVIEW
		#pragma keyword_global_fragment MULTIVIEW

		float4 BloomFragment(Varyings input) : SV_TARGET
		{
			SETUP_INPUT_VIEW_INDEX(input);
			float4 color = SAMPLE_TEXTURE2D_X(_SourceTexture, _SourceTexture_Sampler, input.texcoord);
			return float4(min(max(color.rgb, float3(0, 0, 0)), float3(65504, 65504, 65504)), color.a);
		}
		HLSLEND
	}
	Pass
	{
		Blend One Zero
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex BloomVertex
		#pragma fragment BloomFragment

		#pragma keyword_global_vertex MULTIVIEW
		#pragma keyword_global_fragment MULTIVIEW

		float4 BloomFragment(Varyings input) : SV_TARGET
		{
			SETUP_INPUT_VIEW_INDEX(input);
			float3 color = SAMPLE_TEXTURE2D_X(_SourceTexture, _SourceTexture_Sampler, input.texcoord + float2(-0.5, -0.5) * _TexelSize).rgb;
			color += SAMPLE_TEXTURE2D_X(_SourceTexture, _SourceTexture_Sampler, input.texcoord + float2(-0.5, 0.5) * _TexelSize).rgb;
			color += SAMPLE_TEXTURE2D_X(_SourceTexture, _SourceTexture_Sampler, input.texcoord + float2(0.5, -0.5) * _TexelSize).rgb;
			color += SAMPLE_TEXTURE2D_X(_SourceTexture, _SourceTexture_Sampler, input.texcoord + float2(0.5, 0.5) * _TexelSize).rgb;
			return float4(color * 0.25, 1.0);
		}
		HLSLEND
	}
	Pass
	{
		Blend One Zero
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex BloomVertex
		#pragma fragment BloomFragment

		#pragma keyword_global_vertex MULTIVIEW
		#pragma keyword_global_fragment MULTIVIEW

		TEXTURE2D_X(_SourceAdditionalTexture);	SAMPLER(_SourceAdditionalTexture_Sampler);

		float4 BloomFragment(Varyings input) : SV_TARGET
		{
			SETUP_INPUT_VIEW_INDEX(input);
			float3 color = SAMPLE_TEXTURE2D_X(_SourceTexture, _SourceTexture_Sampler, input.texcoord + float2(-0.5, -0.5) * _TexelSize).rgb;
			color += SAMPLE_TEXTURE2D_X(_SourceTexture, _SourceTexture_Sampler, input.texcoord + float2(-0.5, 0.5) * _TexelSize).rgb;
			color += SAMPLE_TEXTURE2D_X(_SourceTexture, _SourceTexture_Sampler, input.texcoord + float2(0.5, -0.5) * _TexelSize).rgb;
			color += SAMPLE_TEXTURE2D_X(_SourceTexture, _SourceTexture_Sampler, input.texcoord + float2(0.5, 0.5) * _TexelSize).rgb;
			
			float3 additionalColor = SAMPLE_TEXTURE2D_X(_SourceAdditionalTexture, _SourceAdditionalTexture_Sampler, input.texcoord).rgb;
			
			return float4(color * 0.25 + additionalColor * 0.4, 1.0);
		}
		HLSLEND
	}
}