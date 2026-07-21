// ModelQA Skybox背景シェーダー
// 目的:
//   - 画像Skyboxがある場合は2:1 equirectangular画像を表示する。
//   - 画像が無い、または読み込みに失敗した場合は軽量な手続き型環境へ退避する。
//   - 起動速度を落とさないため、HLSLは短く保ち、重い環境生成は行わない。
#include "ModelQAPreviewEnvironment.hlsli"

Texture2D gSkyboxTexture      : register(t5);
SamplerState gMaterialSampler : register(s0);

cbuffer SceneConstants : register(b0)
{
    row_major float4x4 gMVP; // Skybox描画時はInvViewProj
    float4 gBaseColorFactor;
    float4 gEmissiveFactorAndNormalStrength;
    float4 gMaterialParams;
    float4 gFlags;
    float4 gFlags2;
    float4 gCameraExposure; // xyz: camera, w: exposure
    float4 gLightEnv;       // xyz: light dir, w: env intensity
    float4 gTextureCoordSets;
    float4 gLightingModes;  // x: light intensity
    float4 gStyleParams;    // z: skybox preset
};

struct VSOutput
{
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
};

VSOutput VSMain(uint vertexId : SV_VertexID)
{
    float2 p = (vertexId == 0) ? float2(-1.0f, -1.0f) : ((vertexId == 1) ? float2(-1.0f, 3.0f) : float2(3.0f, -1.0f));
    VSOutput o;
    o.position = float4(p, 0.0f, 1.0f);
    o.uv = p * 0.5f + 0.5f;
    return o;
}

float3 ReconstructWorldDirection(float2 uv)
{
    float2 ndc = uv * 2.0f - 1.0f;
    float4 worldFar = mul(float4(ndc, 1.0f, 1.0f), gMVP);
    if (abs(worldFar.w) > 1.0e-6f)
        worldFar.xyz /= worldFar.w;
    float3 dir = worldFar.xyz - gCameraExposure.xyz;
    if (dot(dir, dir) < 1.0e-8f)
        dir = float3(ndc.x, -ndc.y * 0.35f, 1.0f);
    return MQA_SafeNormalize(dir, float3(0.0f, 0.0f, 1.0f));
}

float4 PSMain(VSOutput input) : SV_Target
{
    const float3 dir = ReconstructWorldDirection(input.uv);
    float3 linearColor;

    if (MQA_HasImageSkyboxPreset(gStyleParams.z))
    {
        const float2 skyUv = MQA_SkyboxImageUV(dir);
        if (MQA_IsHdriSkyboxPreset(gStyleParams.z))
            linearColor = max(gSkyboxTexture.SampleLevel(gMaterialSampler, skyUv, 0.0f).rgb, 0.0f) * max(gLightEnv.w, 0.0f);
        else
            linearColor = MQA_SRGBToLinear(gSkyboxTexture.SampleLevel(gMaterialSampler, skyUv, 0.0f).rgb) * max(gLightEnv.w, 0.0f) * 1.05f;
    }
    else
    {
        linearColor = MQA_EvaluateModelQAPreviewEnvironment(
            dir,
            gStyleParams.z,
            gLightEnv.xyz,
            max(gLightEnv.w, 0.0f),
            max(gLightingModes.x, 0.0f),
            0.0f,
            1.0f,
            1.0f);
    }

    linearColor *= max(gCameraExposure.w, 0.01f);
    return float4(MQA_LinearToSRGB(MQA_TonemapReinhard(linearColor)), 1.0f);
}
