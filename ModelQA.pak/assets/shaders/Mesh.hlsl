// ModelQA Mesh/PBRプレビューシェーダー
// 目的:
//   - BaseColor / Normal / MetallicRoughness / AO / Emissive / Glassを軽量に確認する。
//   - 画像Skyboxを反射・屈折にも使い、背景とモデルの映り込みを一致させる。
//   - 初回起動が遅くならないよう、重い分岐や大きい手続き型環境は避ける。
#include "ModelQAPreviewEnvironment.hlsli"

#if defined(MODELQA_BINDLESS)
#include "meshlets/Bindings.hlsli"
#else
Texture2D gBaseColorTexture         : register(t0);
Texture2D gNormalTexture            : register(t1);
Texture2D gMetallicRoughnessTexture : register(t2);
Texture2D gEmissiveTexture          : register(t3);
Texture2D gOcclusionTexture         : register(t4);
Texture2D gSkyboxTexture            : register(t5);
#endif
SamplerState gMaterialSampler       : register(s0);

cbuffer SceneConstants : register(b0)
{
    row_major float4x4 gMVP;
    float4 gBaseColorFactor;
    float4 gEmissiveFactorAndNormalStrength; // xyz: emissive, w: normal strength
    float4 gMaterialParams;                  // x metallic, y roughness, z alphaCutoff, w alphaMode
    uint gMeshFlags;
    float3 gReflectionGridCenter;
    float4 gFlags2;                          // x 0/off,1/UV0,2/UV1 emissive; y 0/off or 1+AO strength
    float4 gCameraExposure;                  // xyz camera, w exposure
    float4 gLightEnv;                        // xyz light dir, w env intensity
    float4 gReflectionGrid;                 // x 細線の間隔、y 主線の間隔、z 片側の範囲
    float4 gLightingModes;                   // x direct, y reflection, z refraction, w IBL
    float4 gStyleParams;                     // x toon, y toon steps, z skybox preset, w glass opacity
};

// SimpleRenderer の MeshDrawFlag と同じ配置。
static const uint MeshBaseColorMap = 1u << 0;
static const uint MeshNormalMap = 1u << 1;
static const uint MeshMetallicRoughnessMap = 1u << 2;
static const uint MeshFlipV = 1u << 3;
static const uint MeshBaseColorUv1 = 1u << 4;
static const uint MeshNormalUv1 = 1u << 5;
static const uint MeshMetallicRoughnessUv1 = 1u << 6;
static const uint MeshOcclusionUv1 = 1u << 7;
static const uint MeshSkyboxVisible = 1u << 8;

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

float2 ChooseUv(VSOutput input, bool useUv1)
{
    float2 uv = useUv1 ? input.uv1 : input.uv0;
    if ((gMeshFlags & MeshFlipV) != 0u)
        uv.y = 1.0f - uv.y;
    return uv;
}

float3 SampleEnv(float3 dir, float roughness)
{
#if defined(MODELQA_BINDLESS)
    Texture2D<float4> gSkyboxTexture = ResourceDescriptorHeap[gSkyboxDescriptor];
#endif
    dir = MQA_SafeNormalize(dir, float3(0.0f, 1.0f, 0.0f));
    float3 result = float3(0.0f, 0.0f, 0.0f);
    if (MQA_HasImageSkyboxPreset(gStyleParams.z))
    {
        const float2 uv = MQA_SkyboxImageUV(dir);
        if (MQA_IsHdriSkyboxPreset(gStyleParams.z))
        {
            const float hdrMip = saturate(roughness) * saturate(roughness) * 12.0f;
            result = max(gSkyboxTexture.SampleLevel(gMaterialSampler, uv, hdrMip).rgb, 0.0f);
        }
        else
        {
            result = MQA_SRGBToLinear(gSkyboxTexture.SampleLevel(gMaterialSampler, uv, 0.0f).rgb);
            // equirect画像にはmipが無い環境もあるため、粗さが高い材質では輝度寄りへ軽くぼかす。
            float l = MQA_Luminance(result);
            result = lerp(result, float3(l, l, l), saturate(roughness) * 0.28f);
        }
        result *= max(gLightEnv.w, 0.0f);
    }
    else
    {
        result = MQA_SampleProceduralEnvironmentRough(dir, roughness, gStyleParams.z, gLightEnv.xyz, max(gLightEnv.w, 0.0f), max(gLightingModes.x, 0.0f), 0.0f, 1.0f, 1.0f);
    }
    return result;
}

float ReflectedGridLine(float2 cells, float blurCells)
{
    const float2 distanceToLine = abs(frac(cells + 0.5f) - 0.5f);
    const float2 filterWidth = max(fwidth(cells), 0.001f) + blurCells;
    const float2 coverage = saturate(0.5f + (0.06f - distanceToLine) / filterWidth) / max(2.0f * filterWidth, 1.0f);
    return max(coverage.x, coverage.y);
}

float3 SampleReflectionScene(float3 origin, float3 direction, float roughness)
{
    float3 result = 0.0f;
    if ((gMeshFlags & MeshSkyboxVisible) != 0u)
        result = SampleEnv(direction, roughness);
    else if (gReflectionGrid.x > 0.0f)
    {
        // 黒背景では、レイが有限のグリッド面と交差した場所だけに反射光を返す。
        const float vertical = (direction.y < 0.0f ? -1.0f : 1.0f) * max(abs(direction.y), 0.0001f);
        const float distance = (gReflectionGridCenter.y - origin.y) / vertical;
        const float2 offset = origin.xz - gReflectionGridCenter.xz + distance * direction.xz;
        const float blur = roughness * roughness * abs(distance) * 0.25f;
        const float minor = ReflectedGridLine(offset / gReflectionGrid.x, blur / gReflectionGrid.x);
        const float major = ReflectedGridLine(offset / gReflectionGrid.y, blur / gReflectionGrid.y);
        const bool visible = abs(direction.y) >= 0.0001f && distance > 0.0f && all(abs(offset) <= gReflectionGrid.zz);
        if (visible)
            result = lerp(minor * float3(0.042f, 0.046f, 0.052f), float3(0.175f, 0.187f, 0.207f), major);
    }
    return result;
}

float4 BuildNormal(VSOutput input, float3 N)
{
#if defined(MODELQA_BINDLESS)
    Texture2D<float4> gNormalTexture = ResourceDescriptorHeap[gNormalDescriptor];
#endif
    const bool normalUsesUv1 = (gMeshFlags & MeshNormalUv1) != 0u;
    const float2 uv = ChooseUv(input, normalUsesUv1);
    // 接線の有無がピクセル間で異なっても微分を壊さないよう、分岐の外で求める。
    const float3 positionDx = ddx(input.worldPos);
    const float3 positionDy = ddy(input.worldPos);
    const float2 uvDx = ddx(uv);
    const float2 uvDy = ddy(uv);
    float4 result = float4(N, 0.0f);
    if ((gMeshFlags & MeshNormalMap) != 0u && gEmissiveFactorAndNormalStrength.w > 0.0001f)
    {
        float3 T = input.tangent.xyz;
        float3 B = float3(0.0f, 0.0f, 0.0f);
        if (dot(T, T) >= 1.0e-6f && !normalUsesUv1)
        {
            float handedness = (abs(input.tangent.w) > 0.001f) ? input.tangent.w : 1.0f;
            B = cross(N, T) * handedness * ((gMeshFlags & MeshFlipV) != 0u ? -1.0f : 1.0f);
        }
        else
        {
            // 表示専用の基底。元TANGENTや解析用の属性は生成・変更しない。
            const float determinant = uvDx.x * uvDy.y - uvDx.y * uvDy.x;
            const float uvScaleSquared = max(dot(uvDx, uvDx), dot(uvDy, uvDy));
            const bool validUvBasis = uvScaleSquared > 0.0f && abs(determinant) > uvScaleSquared * 1.0e-6f;
            const float orientation = validUvBasis ? (determinant < 0.0f ? -1.0f : 1.0f) : 0.0f;
            T = (positionDx * uvDy.y - positionDy * uvDx.y) * orientation;
            B = (positionDy * uvDx.x - positionDx * uvDy.x) * orientation;
        }
        const float tangentScale = max(abs(T.x), max(abs(T.y), abs(T.z)));
        const float bitangentScale = max(abs(B.x), max(abs(B.y), abs(B.z)));
        if (all(isfinite(T)) && all(isfinite(B)) && tangentScale > 0.0f && bitangentScale > 0.0f)
        {
            T /= tangentScale;
            B /= bitangentScale;
            T -= N * dot(N, T);
            if (dot(T, T) > 1.0e-8f)
            {
                T = normalize(T);
                B = cross(N, T) * (dot(cross(N, T), B) < 0.0f ? -1.0f : 1.0f);
                float3 nt = gNormalTexture.SampleGrad(gMaterialSampler, uv, uvDx, uvDy).xyz * 2.0f - 1.0f;
                // ミップで平均化された法線の長さから、失われた方向のばらつきを粗さに回す。
                const float normalVariance = saturate(1.0f - dot(nt, nt)) * saturate(gEmissiveFactorAndNormalStrength.w);
                nt.xy *= gEmissiveFactorAndNormalStrength.w;
                result = float4(MQA_SafeNormalize(T * nt.x + B * nt.y + N * nt.z, N), normalVariance);
            }
        }
    }
    return result;
}

float3 Fresnel(float cosTheta, float3 f0)
{
    float f = pow(1.0f - saturate(cosTheta), 5.0f);
    return f0 + (1.0f - f0) * f;
}

float3 EvaluateDirectPbr(float3 diffuseColor, float3 f0, float roughness,
    float NdotL, float NdotV, float NdotH, float VdotH)
{
    const float invPi = 0.318309886f;
    const float alpha = roughness * roughness;
    const float alphaSquared = alpha * alpha;
    // 分母を sin² と cos² に分け、鏡面付近で値が桁落ちするのを防ぐ。
    const float denominator = (1.0f - NdotH * NdotH) + alphaSquared * NdotH * NdotH;
    const float distribution = alphaSquared * invPi / (denominator * denominator);
    const float viewMask = NdotL * sqrt(NdotV * NdotV * (1.0f - alphaSquared) + alphaSquared);
    const float lightMask = NdotV * sqrt(NdotL * NdotL * (1.0f - alphaSquared) + alphaSquared);
    const float visibility = 0.5f / max(viewMask + lightMask, 1.0e-5f);
    const float3 fresnel = Fresnel(VdotH, f0);
    return ((1.0f - fresnel) * diffuseColor * invPi + fresnel * distribution * visibility) * NdotL;
}

float3 SampleDiffuseEnvironment(float3 N)
{
    // 半球の中心と斜め4方向を使い、上空・地面の照明を面の向きに応じて受ける。
    const float3 axis = abs(N.y) < 0.99f ? float3(0, 1, 0) : float3(1, 0, 0);
    const float3 tangent = normalize(cross(axis, N));
    const float3 bitangent = cross(N, tangent);
    const float3 center = 0.70710678f * N;
    const float3 offsetX = 0.70710678f * tangent;
    const float3 offsetY = 0.70710678f * bitangent;
    return SampleEnv(N, 0.78f) * 0.4f + 0.15f *
        (SampleEnv(center + offsetX, 0.78f) + SampleEnv(center - offsetX, 0.78f) +
         SampleEnv(center + offsetY, 0.78f) + SampleEnv(center - offsetY, 0.78f));
}

float4 PSMain(VSOutput input) : SV_Target
{
    // Negative debug-mode value is reserved for the unlit wire pass.
    // Return before material alpha/vertex colors can change the requested alpha.
    if (gFlags2.z < -0.5f) return gBaseColorFactor;
#if defined(MODELQA_BINDLESS)
    Texture2D<float4> gBaseColorTexture = ResourceDescriptorHeap[gBaseColorDescriptor];
    Texture2D<float4> gMetallicRoughnessTexture = ResourceDescriptorHeap[gMetallicRoughnessDescriptor];
    Texture2D<float4> gOcclusionTexture = ResourceDescriptorHeap[gOcclusionDescriptor];
    Texture2D<float4> gEmissiveTexture = ResourceDescriptorHeap[gEmissiveDescriptor];
#endif
    float4 baseTex = ((gMeshFlags & MeshBaseColorMap) != 0u)
        ? gBaseColorTexture.Sample(gMaterialSampler, ChooseUv(input, (gMeshFlags & MeshBaseColorUv1) != 0u)) : 1.0f;
    float3 baseColor = MQA_SRGBToLinear(baseTex.rgb) * gBaseColorFactor.rgb * input.color.rgb;
    float alpha = baseTex.a * gBaseColorFactor.a * input.color.a;

    if (gMaterialParams.w < 0.5f)
        alpha = 1.0f;
    else if (gMaterialParams.w < 1.5f)
    {
        clip(alpha - gMaterialParams.z);
        alpha = 1.0f;
    }

    float3 V = MQA_SafeNormalize(gCameraExposure.xyz - input.worldPos, float3(0.0f, 0.0f, 1.0f));
    float3 L = MQA_SafeNormalize(gLightEnv.xyz, float3(-0.35f, 0.78f, -0.52f));
    const float3 geometricNormal = cross(ddx(input.worldPos), ddy(input.worldPos));
    const float3 fallbackNormal = MQA_SafeNormalize(
        dot(geometricNormal, V) < 0.0f ? -geometricNormal : geometricNormal, float3(0.0f, 1.0f, 0.0f));
    const float3 surfaceNormal = MQA_SafeNormalize(input.normal, fallbackNormal);
    const float4 mappedNormal = BuildNormal(input, surfaceNormal);
    const float3 N = mappedNormal.xyz;

    float metallic = saturate(gMaterialParams.x);
    float roughness = saturate(gMaterialParams.y);
    if ((gMeshFlags & MeshMetallicRoughnessMap) != 0u)
    {
        float4 mr = gMetallicRoughnessTexture.Sample(gMaterialSampler, ChooseUv(input, (gMeshFlags & MeshMetallicRoughnessUv1) != 0u));
        roughness *= mr.g;
        metallic *= mr.b;
    }
    roughness = clamp(roughness, 0.08f, 1.0f);

    float ao = 1.0f;
    if (gFlags2.y > 0.5f)
        ao = lerp(1.0f, gOcclusionTexture.Sample(gMaterialSampler, ChooseUv(input, (gMeshFlags & MeshOcclusionUv1) != 0u)).r, saturate(gFlags2.y - 1.0f));

    float3 emissive = gEmissiveFactorAndNormalStrength.rgb;
    if (gFlags2.x > 0.5f)
        emissive *= MQA_SRGBToLinear(gEmissiveTexture.Sample(gMaterialSampler, ChooseUv(input, gFlags2.x > 1.5f)).rgb);

    float NdotL = saturate(dot(N, L));
    float NdotV = saturate(dot(N, V));
    float3 H = MQA_SafeNormalize(L + V, L);
    float NdotH = saturate(dot(N, H));

    // 編集用の表示モデル。保存する標準PBR材質とは独立して比較する。
    if (gFlags2.z > 1.5f && gFlags2.z < 2.5f)
        return float4(MQA_LinearToSRGB(baseColor + emissive), alpha);
    if (gFlags2.z > 2.5f && gFlags2.z < 4.5f)
    {
        float3 shaded = baseColor * (0.15f + NdotL * max(gLightingModes.x, 0.0f)) * ao + emissive;
        if (gFlags2.z > 3.5f)
            shaded += pow(NdotH, lerp(128.0f, 4.0f, roughness)) * max(gLightingModes.x, 0.0f);
        return float4(MQA_LinearToSRGB(MQA_TonemapReinhard(shaded * max(gCameraExposure.w, 0.01f))), alpha);
    }

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

    roughness = sqrt(saturate(roughness * roughness + 0.35f * mappedNormal.w));
    const float3 diffuseColor = baseColor * (1.0f - metallic);
    const float VdotH = saturate(dot(V, H));
    const float3 direct = EvaluateDirectPbr(diffuseColor, f0, roughness, NdotL, NdotV, NdotH, VdotH)
        * max(gLightingModes.x, 0.0f);

    float3 R = reflect(-V, N);
    const float grazing = pow(1.0f - NdotV, 5.0f);
    const float3 envFresnel = f0 + (max(1.0f - roughness, f0) - f0) * grazing;
    const float ibl = max(gLightingModes.w, 0.0f);
    float3 envDiffuse = 0.0f;
    if (ibl > 0.0f && metallic < 1.0f)
        envDiffuse = SampleDiffuseEnvironment(N) * diffuseColor * (1.0f - envFresnel) * ibl;
    const float3 sceneReflection = SampleReflectionScene(input.worldPos, R, roughness) * max(gLightingModes.y, 0.0f);
    float3 envSpec = sceneReflection * envFresnel;

    float3 color = direct + (envDiffuse + envSpec) * ao + emissive;

    if (gFlags2.w > 0.5f)
    {
        float3 T = refract(-V, N, 1.0f / 1.45f);
        T = (dot(T, T) > 1.0e-6f) ? MQA_SafeNormalize(T, R) : R;
        float3 refracted = SampleReflectionScene(input.worldPos, T, min(1.0f, roughness + 0.22f)) * max(gLightingModes.z, 0.0f);
        float fres = saturate(0.04f + 0.96f * pow(1.0f - NdotV, 5.0f));
        float3 tint = lerp(float3(0.88f, 0.96f, 1.0f), max(baseColor, 0.05f), 0.15f);
        color = lerp(refracted * tint, sceneReflection, fres);
    }

    if (gStyleParams.x > 0.5f)
    {
        float steps = max(gStyleParams.y, 2.0f);
        float band = floor(NdotL * steps) / max(steps - 1.0f, 1.0f);
        color = baseColor * (0.18f + band * 0.82f) + envSpec * 0.28f + emissive;
    }

    return float4(MQA_LinearToSRGB(MQA_TonemapReinhard(color * max(gCameraExposure.w, 0.01f))), alpha);
}
