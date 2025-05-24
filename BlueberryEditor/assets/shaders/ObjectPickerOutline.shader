Shader
{
	Pass
	{
		Blend SrcAlpha OneMinusSrcAlpha
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex ObjectPickerOutlineVertex
		#pragma fragment ObjectPickerOutlineFragment

		#include "Core.hlsl"

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

		Varyings ObjectPickerOutlineVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = float4(input.positionOS, 1.0f);
			output.texcoord = input.texcoord;
			return output;
		}

		TEXTURE2D(_PickingTexture);		SAMPLER(_PickingTexture_Sampler);
		
		float4 ObjectPickerOutlineFragment(Varyings input) : SV_TARGET
		{
			float2 offset = float2(3.0 * RENDER_TARGET_SIZE_INV_SIZE.z, 3.0 * RENDER_TARGET_SIZE_INV_SIZE.w);

			float4 sample1 = SAMPLE_TEXTURE2D(_PickingTexture, _PickingTexture_Sampler, input.texcoord);

			float4 sample2 = SAMPLE_TEXTURE2D(_PickingTexture, _PickingTexture_Sampler, input.texcoord + float2(offset.x, 0));
			float4 sample3 = SAMPLE_TEXTURE2D(_PickingTexture, _PickingTexture_Sampler, input.texcoord - float2(offset.x, 0));
			float4 sample4 = SAMPLE_TEXTURE2D(_PickingTexture, _PickingTexture_Sampler, input.texcoord + float2(0, offset.y));
			float4 sample5 = SAMPLE_TEXTURE2D(_PickingTexture, _PickingTexture_Sampler, input.texcoord - float2(0, offset.y));
			float4 sample6 = SAMPLE_TEXTURE2D(_PickingTexture, _PickingTexture_Sampler, input.texcoord + float2(offset.x, offset.y));
			float4 sample7 = SAMPLE_TEXTURE2D(_PickingTexture, _PickingTexture_Sampler, input.texcoord - float2(offset.x, offset.y));
			float4 sample8 = SAMPLE_TEXTURE2D(_PickingTexture, _PickingTexture_Sampler, input.texcoord + float2(offset.x, -offset.y));
			float4 sample9 = SAMPLE_TEXTURE2D(_PickingTexture, _PickingTexture_Sampler, input.texcoord - float2(offset.x, -offset.y));

			if (sample1.r == 0)
			{
				return float4(1, 165.0 / 255.0, 0, 1) *
					(
						sample2.r > 0 ||
						sample3.r > 0 ||
						sample4.r > 0 ||
						sample5.r > 0 ||
						sample6.r > 0 ||
						sample7.r > 0 ||
						sample8.r > 0 ||
						sample9.r > 0
					);
			}

			return float4(0, 0, 0, 0);
		}
		HLSLEND
	}
}