Shader
{
	Pass
	{
		Blend SrcAlpha OneMinusSrcAlpha One Zero
		ZWrite Off
		ZTest Always
		Cull None

		HLSLBEGIN
		#pragma vertex UiVertex
		#pragma fragment UiFragment

		#include "Core.hlsl"

		struct Attributes
		{
			float2 positionOS : POSITION;
			float2 texcoord : TEXCOORD0;
			float4 color : COLOR;
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			float2 texcoord : TEXCOORD0;
			float4 color : COLOR;
		};

		Varyings UiVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = mul(VIEW_PROJECTION_MATRIX, mul(_ModelMatrix, float4(input.positionOS, 0.0f, 1.0f)));
			output.texcoord = input.texcoord;
			output.color = input.color;
			return output;
		}

		float4 UiFragment(Varyings input) : SV_TARGET
		{
			return input.color;
		}
		HLSLEND
	}
	Pass
	{
		Blend SrcAlpha OneMinusSrcAlpha One OneMinusSrcAlpha
		ZWrite Off
		ZTest Always
		Cull None

		HLSLBEGIN
		#pragma vertex UiVertex
		#pragma fragment UiFragment

		#include "Core.hlsl"

		struct Attributes
		{
			float2 positionOS : POSITION;
			float2 texcoord : TEXCOORD0;
			float4 color : COLOR;
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			float2 texcoord : TEXCOORD0;
			float4 color : COLOR;
		};

		Varyings UiVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = mul(VIEW_PROJECTION_MATRIX, mul(_ModelMatrix, float4(input.positionOS, 0.0f, 1.0f)));
			output.texcoord = input.texcoord;
			output.color = input.color;
			return output;
		}

		TEXTURE2D(_UiTexture);	SAMPLER(_UiTexture_Sampler);

		float4 UiFragment(Varyings input) : SV_TARGET
		{
			float4 color = SAMPLE_TEXTURE2D(_UiTexture, _UiTexture_Sampler, input.texcoord);
			return input.color * color;
		}
		HLSLEND
	}
}