// ModelQA Mesh/PBRプレビューシェーダー
// 目的:
//   - BaseColor / Normal / MetallicRoughness / AO / Emissive / Glassを軽量に確認する。
//   - 画像Skyboxを反射・屈折にも使い、背景とモデルの映り込みを一致させる。
//   - 初回起動が遅くならないよう、重い分岐や大きい手続き型環境は避ける。
#include "ModelQAPreviewEnvironment.hlsli"

Texture2D gBaseColorTexture         : register(t0);
Texture2D gNormalTexture            : register(t1);
Texture2D gMetallicRoughnessTexture : register(t2);
Texture2D gEmissiveTexture          : register(t3);
Texture2D gOcclusionTexture         : register(t4);
Texture2D gSkyboxTexture            : register(t5);
SamplerState gMaterialSampler       : register(s0);

cbuffer SceneConstants : register(b0)
{
    row_major float4x4 gMVP;
    float4 gBaseColorFactor;
    float4 gEmissiveFactorAndNormalStrength; // xyz: emissive, w: normal strength
    float4 gMaterialParams;                  // x metallic, y roughness, z alphaCutoff, w alphaMode
    float4 gFlags;                           // x base, y normal, z MR, w flipV
    float4 gFlags2;                          // x emissive, y AO, z richPreview, w glassLike
    float4 gCameraExposure;                  // xyz camera, w exposure
    float4 gLightEnv;                        // xyz light dir, w env intensity
    float4 gTextureCoordSets;                // x base, y normal, z MR, w AO
    float4 gLightingModes;                   // x direct, y reflection, z refraction, w IBL
    float4 gStyleParams;                     // x toon, y toon steps, z skybox preset, w glass opacity
};

struct VSInput
{
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float4 tangent  : TANGENT;
    float2 uv0      : TEXCOORD0;
    float2 uv1      : TEXCOORD1;
    float4 color    : COLOR;
};

struct VSOutput
{
    float4 position : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 normal   : TEXCOORD1;
    float4 tangent  : TEXCOORD2;
    float2 uv0      : TEXCOORD3;
    float2 uv1      : TEXCOORD4;
    float4 color    : COLOR;
};

VSOutput VSMain(VSInput input)
{
    VSOutput o;
    o.position = mul(float4(input.position, 1.0f), gMVP);
    o.worldPos = input.position;
    o.normal = input.normal;
    o.tangent = input.tangent;
    o.uv0 = input.uv0;
    o.uv1 = input.uv1;
    o.color = input.color;
    return o;
}

float2 ChooseUv(VSOutput input, float texCoord)
{
    float2 uv = (texCoord > 0.5f) ? input.uv1 : input.uv0;
    if (gFlags.w > 0.5f)
        uv.y = 1.0f - uv.y;
    return uv;
}

float3 SampleEnv(float3 dir, float roughness)
{
    dir = MQA_SafeNormalize(dir, float3(0.0f, 1.0f, 0.0f));
    float3 c;
    if (MQA_HasImageSkyboxPreset(gStyleParams.z))
    {
        c = MQA_SRGBToLinear(gSkyboxTexture.SampleLevel(gMaterialSampler, MQA_SkyboxImageUV(dir), 0.0f).rgb);
        // equirect画像にはmipが無い環境もあるため、粗さが高い材質では輝度寄りへ軽くぼかす。
        float l = MQA_Luminance(c);
        c = lerp(c, float3(l, l, l), saturate(roughness) * 0.28f);
        return c * max(gLightEnv.w, 0.0f);
    }
    return MQA_SampleProceduralEnvironmentRough(dir, roughness, gStyleParams.z, gLightEnv.xyz, max(gLightEnv.w, 0.0f), max(gLightingModes.x, 0.0f), 0.0f, 1.0f, 1.0f);
}

float3 BuildNormal(VSOutput input, float3 N)
{
    if (gFlags.y < 0.5f || gEmissiveFactorAndNormalStrength.w <= 0.0001f)
        return N;
    float3 T = input.tangent.xyz;
    if (dot(T, T) < 1.0e-6f)
        return N;
    T = MQA_SafeNormalize(T - N * dot(N, T), float3(1.0f, 0.0f, 0.0f));
    float handedness = (abs(input.tangent.w) > 0.001f) ? input.tangent.w : 1.0f;
    float3 B = MQA_SafeNormalize(cross(N, T) * handedness, float3(0.0f, 0.0f, 1.0f));
    float3 nt = gNormalTexture.Sample(gMaterialSampler, ChooseUv(input, gTextureCoordSets.y)).xyz * 2.0f - 1.0f;
    nt.xy *= saturate(gEmissiveFactorAndNormalStrength.w);
    return MQA_SafeNormalize(T * nt.x + B * nt.y + N * nt.z, N);
}

float3 Fresnel(float cosTheta, float3 f0)
{
    float f = pow(1.0f - saturate(cosTheta), 5.0f);
    return f0 + (1.0f - f0) * f;
}

float4 PSMain(VSOutput input) : SV_Target
{
    float4 baseTex = (gFlags.x > 0.5f) ? gBaseColorTexture.Sample(gMaterialSampler, ChooseUv(input, gTextureCoordSets.x)) : float4(1.0f, 1.0f, 1.0f, 1.0f);
    float3 baseColor = MQA_SRGBToLinear(baseTex.rgb) * gBaseColorFactor.rgb * input.color.rgb;
    float alpha = baseTex.a * gBaseColorFactor.a * input.color.a;

    if (gMaterialParams.w < 0.5f)
        alpha = 1.0f;
    else if (gMaterialParams.w < 1.5f)
    {
        clip(alpha - gMaterialParams.z);
        alpha = 1.0f;
    }

    float3 N = MQA_SafeNormalize(input.normal, float3(0.0f, 1.0f, 0.0f));
    N = BuildNormal(input, N);
    float3 V = MQA_SafeNormalize(gCameraExposure.xyz - input.worldPos, float3(0.0f, 0.0f, 1.0f));
    float3 L = MQA_SafeNormalize(gLightEnv.xyz, float3(-0.35f, 0.78f, -0.52f));

    float metallic = saturate(gMaterialParams.x);
    float roughness = saturate(gMaterialParams.y);
    if (gFlags.z > 0.5f)
    {
        float4 mr = gMetallicRoughnessTexture.Sample(gMaterialSampler, ChooseUv(input, gTextureCoordSets.z));
        roughness *= mr.g;
        metallic *= mr.b;
    }
    roughness = clamp(roughness, 0.08f, 1.0f);

    float ao = 1.0f;
    if (gFlags2.y > 0.5f)
        ao = lerp(1.0f, gOcclusionTexture.Sample(gMaterialSampler, ChooseUv(input, gTextureCoordSets.w)).r, 0.85f);

    float3 emissive = gEmissiveFactorAndNormalStrength.rgb;
    if (gFlags2.x > 0.5f)
        emissive *= MQA_SRGBToLinear(gEmissiveTexture.Sample(gMaterialSampler, ChooseUv(input, gTextureCoordSets.x)).rgb);

    float NdotL = saturate(dot(N, L));
    float NdotV = saturate(dot(N, V));
    float3 H = MQA_SafeNormalize(L + V, L);
    float NdotH = saturate(dot(N, H));

    if (gFlags2.z < 0.5f)
    {
        float shade = 0.18f * max(gLightingModes.w, 0.0f) + NdotL * max(gLightingModes.x, 0.0f);
        float3 simpleColor = baseColor * max(shade, 0.05f) * ao + emissive;
        return float4(MQA_LinearToSRGB(MQA_TonemapReinhard(simpleColor * max(gCameraExposure.w, 0.01f))), alpha);
    }

    float3 f0 = lerp(float3(0.04f, 0.04f, 0.04f), baseColor, metallic);
    if (gFlags2.w > 0.5f)
    {
        metallic = 0.0f;
        roughness = min(roughness, 0.18f);
        f0 = float3(0.04f, 0.045f, 0.05f);
        alpha = min(alpha, saturate(gStyleParams.w));
    }

    float3 F = Fresnel(NdotV, f0);
    float specPower = lerp(80.0f, 8.0f, roughness);
    float spec = pow(NdotH, specPower) * (1.0f - roughness * 0.55f);
    float3 direct = (baseColor * (1.0f - metallic) * (0.12f + NdotL) + F * spec) * max(gLightingModes.x, 0.0f);

    float3 R = reflect(-V, N);
    float3 envDiffuse = SampleEnv(N, 1.0f) * baseColor * (1.0f - metallic) * max(gLightingModes.w, 0.0f);
    float3 envSpec = SampleEnv(R, roughness) * F * max(gLightingModes.y, 0.0f);
    if (gFlags2.w < 0.5f)
    {
        float l = MQA_Luminance(envSpec);
        envSpec = lerp(envSpec, float3(l, l, l), roughness * 0.42f);
    }

    float3 color = (direct + envDiffuse + envSpec) * ao + emissive;

    if (gFlags2.w > 0.5f)
    {
        float3 T = refract(-V, N, 1.0f / 1.45f);
        T = (dot(T, T) > 1.0e-6f) ? MQA_SafeNormalize(T, R) : R;
        float3 reflected = SampleEnv(R, roughness) * max(gLightingModes.y, 0.0f);
        float3 refracted = SampleEnv(T, min(1.0f, roughness + 0.22f)) * max(gLightingModes.z, 0.0f);
        float fres = saturate(0.04f + 0.96f * pow(1.0f - NdotV, 5.0f));
        float3 tint = lerp(float3(0.88f, 0.96f, 1.0f), max(baseColor, 0.05f), 0.15f);
        color = lerp(refracted * tint, reflected, fres);
    }

    if (gStyleParams.x > 0.5f)
    {
        float steps = max(gStyleParams.y, 2.0f);
        float band = floor(NdotL * steps) / max(steps - 1.0f, 1.0f);
        color = baseColor * (0.18f + band * 0.82f) + envSpec * 0.28f + emissive;
    }

    return float4(MQA_LinearToSRGB(MQA_TonemapReinhard(color * max(gCameraExposure.w, 0.01f))), alpha);
}
