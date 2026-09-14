Shader
{
	Pass
	{
		Blend One Zero
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex FxaaVertex
		#pragma fragment FxaaFragment

		#include "Core.hlsl"

		#define FXAA_PC 1
		#define FXAA_HLSL_5 1
		#define FXAA_EARLY_EXIT 0
		#include "Fxaa3_11.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
			float2 texcoord : TEXCOORD0;
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			float2 texcoord : TEXCOORD0;
		};

		TEXTURE2D(_SourceTexture);	SAMPLER(_SourceTexture_Sampler);

		Varyings FxaaVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = float4(input.positionOS, 1.0f);
			output.texcoord = input.texcoord * (CAMERA_SIZE_INV_SIZE.xy * RENDER_TARGET_SIZE_INV_SIZE.zw);
			return output;
		}

		float4 FxaaFragment(Varyings input) : SV_TARGET
		{
			FxaaTex sourceTex;
			sourceTex.smpl = _SourceTexture_Sampler;
			sourceTex.tex = _SourceTexture;
			FxaaTex emptyTex;
			return float4(FxaaPixelShader(input.texcoord, float4(0, 0, 0, 0), sourceTex, emptyTex, emptyTex, RENDER_TARGET_SIZE_INV_SIZE.zw, float4(0, 0, 0, 0), float4(0, 0, 0, 0), float4(0, 0, 0, 0), 0.75, 0.166, 0.0625, 0, 0, 0, float4(0, 0, 0, 0)).rgb, 1.0);
		}
		HLSLEND
	}
}