Shader
{
	Pass
	{
		Blend One Zero
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex DeferredVertex
		#pragma fragment DeferredFragment
		
		#pragma keyword_global_fragment SHADOWS
		#pragma keyword_global_fragment REFLECTIONS

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

		TEXTURE2D(_ScreenColorTexture);	SAMPLER(_ScreenColorTexture_Sampler);
		TEXTURE2D(_ScreenDepthStencilTexture);
		TEXTURE2D(_ScreenNormalWSTexture);
		TEXTURE2D(_ScreenORMTexture);
		TEXTURE2D(_ScreenBakedGITexture);

		Varyings DeferredVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = float4(input.positionOS, 1.0f);
			output.texcoord = input.texcoord;
			return output;
		}

		float4 DeferredFragment(Varyings input) : SV_TARGET
		{
			float2 uv = GetRenderTargetUV(input.positionCS);
			float3 albedo = SAMPLE_TEXTURE2D(_ScreenColorTexture, _ScreenColorTexture_Sampler, uv).rgb;
			float depth = SAMPLE_TEXTURE2D(_ScreenDepthStencilTexture, _ScreenColorTexture_Sampler, uv).r;
			float3 normalWS = DecodeNormalOctahedral(SAMPLE_TEXTURE2D(_ScreenNormalWSTexture, _ScreenColorTexture_Sampler, uv).rg);
			float3 orm = SAMPLE_TEXTURE2D(_ScreenORMTexture, _ScreenColorTexture_Sampler, uv).rgb;
			float3 bakedGI = SAMPLE_TEXTURE2D(_ScreenBakedGITexture, _ScreenColorTexture_Sampler, uv).rgb;

			float3 positionCS = float3(float2(input.texcoord.x, 1 - input.texcoord.y) * 2 - 1, depth);
			float3 positionVS = TransformClipToView(positionCS);
			float3 positionWS = TransformClipToWorld(positionCS);
			float3 viewDirectionWS = GetNormalizedViewDirectionWS(positionWS);

			SurfaceData surfaceData;
			surfaceData.albedo = albedo;
			surfaceData.alpha = 1.0;
			surfaceData.metallic = orm.b;
			surfaceData.roughness = orm.g;
			surfaceData.emission = 0;
			surfaceData.occlusion = orm.r;

			InputData inputData;
			inputData.positionWS = positionWS;
			inputData.positionVS = positionVS;
			inputData.normalWS = normalWS;
			inputData.normalGS = normalWS;
			inputData.normalizedScreenSpaceUV = input.texcoord;
			inputData.renderTargetUV = uv;
			inputData.viewDirectionWS = viewDirectionWS;
			inputData.bakedGI = bakedGI;

			float4 color = float4(CalculatePBR(surfaceData, inputData), surfaceData.alpha);
			color = ApplyVolumetricFog(color, input.texcoord, depth);
			return color;
		}
		HLSLEND
	}
}