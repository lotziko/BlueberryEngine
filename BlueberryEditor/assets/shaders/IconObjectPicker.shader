Shader
{
	Pass
	{
		Blend One Zero
		ZWrite On
		Cull None

		HLSLBEGIN
		#pragma vertex IconObjectPickerVertex
		#pragma fragment IconObjectPickerFragment

		#include "Core.hlsl"

		cbuffer PerObjectData
		{
			float4 _ObjectId;
		}

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

		Varyings IconObjectPickerVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = mul(mul(float4(input.positionOS, 1.0f), _ModelMatrix), VIEW_PROJECTION_MATRIX);
			output.texcoord = input.texcoord;
			return output;
		}

		TEXTURE2D(_BaseMap);	SAMPLER(_BaseMap_Sampler);

		float4 IconObjectPickerFragment(Varyings input) : SV_TARGET
		{
			float4 color = SAMPLE_TEXTURE2D(_BaseMap, _BaseMap_Sampler, input.texcoord);
			clip(color.a - 0.125);
			return float4(_ObjectId.r, _ObjectId.g, 0, 1);
		}
		HLSLEND
	}
}