#include "MetricSummary.hlsli"

StructuredBuffer<float3> Positions : register(t0);
StructuredBuffer<uint> Indices : register(t1);
StructuredBuffer<uint> TrianglePresent : register(t2);
RWStructuredBuffer<float> TriangleArea : register(u0);
RWStructuredBuffer<uint> Candidates : register(u1);
RWStructuredBuffer<MetricSummary> GroupSummaries : register(u2);
RWByteAddressBuffer CandidateCounters : register(u3); // candidate count, omitted (8 bytes)
cbuffer GeometryConstants : register(b0)
{
    uint ElementCount;
    uint CandidateCapacity;
    float Threshold;
    float HistogramUpper;
};
groupshared MetricStatistics SharedStats[256];
groupshared uint SharedHistogram[64];
groupshared uint SharedCandidateCount;
groupshared uint SharedCandidateBase;

[numthreads(1,1,1)]
void CSClear(uint3 tid : SV_DispatchThreadID)
{
    uint bytes; CandidateCounters.GetDimensions(bytes);
    if (tid.x == 0 && tid.y == 0 && tid.z == 0 && bytes >= 8)
    { CandidateCounters.Store(0,0); CandidateCounters.Store(4,0); }
}

// SM 5.1 ABI is documented in GpuAssetAnalysis.hpp. Compile with strict IEEE
// behavior because non-finite source attributes are part of the QA result.
[numthreads(256,1,1)]
void CSMain(uint3 tid : SV_DispatchThreadID, uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    if (lane < 64) SharedHistogram[lane] = 0;
    if (lane == 0) { SharedCandidateCount=0; SharedCandidateBase=0xffffffffu; }
    GroupMemoryBarrierWithGroupSync();

    uint positionCount,positionStride,indexCount,indexStride,presentCount,presentStride;
    uint areaCount,areaStride,candidateCount,candidateStride,summaryCount,summaryStride,counterBytes;
    Positions.GetDimensions(positionCount,positionStride);
    Indices.GetDimensions(indexCount,indexStride);
    TrianglePresent.GetDimensions(presentCount,presentStride);
    TriangleArea.GetDimensions(areaCount,areaStride);
    Candidates.GetDimensions(candidateCount,candidateStride);
    GroupSummaries.GetDimensions(summaryCount,summaryStride);
    CandidateCounters.GetDimensions(counterBytes);
    const uint capacity = min(min(CandidateCapacity,candidateCount),4096u);
    const uint requiredIndices = ElementCount <= 0x55555555u ? ElementCount*3u : 0xffffffffu;
    const bool configurationValid = ElementCount <= 65535u*256u &&
        requiredIndices <= indexCount && ElementCount <= presentCount && ElementCount <= areaCount &&
        summaryCount >= ElementCount/256u + (ElementCount%256u != 0 ? 1u : 0u) &&
        positionStride == 12 && indexStride == 4 && presentStride == 4 && areaStride == 4 &&
        candidateStride == 4 && summaryStride == 288 && counterBytes >= 8 &&
        MetricFinite(Threshold) && Threshold >= 0 && MetricFinite(HistogramUpper) && HistogramUpper > 0;
    const bool active = tid.x < ElementCount;
    MetricStatistics stats = (MetricStatistics)0;
    float area = asfloat(0x7fc00000u);
    uint localCandidate = 0;
    if (active)
    {
        stats.totalCount = 1;
        bool valid = configurationValid;
        bool present = true;
        if (configurationValid) present = TrianglePresent[tid.x] != 0;
        if (!present) stats.skippedCount = 1;
        else
        {
            if (valid)
            {
                const uint base = tid.x*3u;
                const uint3 vertex = uint3(Indices[base],Indices[base+1u],Indices[base+2u]);
                valid = all(vertex < positionCount);
                if (valid)
                {
                    const float3 p0=Positions[vertex.x], p1=Positions[vertex.y], p2=Positions[vertex.z];
                    valid = all((asuint(p0) & 0x7f800000u) != 0x7f800000u) &&
                        all((asuint(p1) & 0x7f800000u) != 0x7f800000u) &&
                        all((asuint(p2) & 0x7f800000u) != 0x7f800000u);
                    if (valid)
                    {
                        precise float3 crossValue = cross(p1-p0,p2-p0);
                        precise float scale = max(abs(crossValue.x),max(abs(crossValue.y),abs(crossValue.z)));
                        if (scale == 0) area = 0;
                        else
                        {
                            precise float3 scaled = crossValue/scale;
                            area = 0.5f*scale*sqrt(dot(scaled,scaled));
                        }
                        valid = MetricFinite(area);
                    }
                }
            }
            if (valid)
            {
                stats.finiteCount=1; stats.minimum=area; stats.maximum=area; stats.average=area;
                const uint bin = area >= HistogramUpper ? 63u : min((uint)((area/HistogramUpper)*63.0f),62u);
                InterlockedAdd(SharedHistogram[bin],1);
            }
            else { stats.invalidCount=1; area=asfloat(0x7fc00000u); }
            // Degenerate/small-area candidates include a conservative boundary
            // band. CPU geometry predicates make the final classification.
            if (!valid || area <= Threshold+0.00001f*max(area,1.0f))
            {
                stats.candidateCount=1;
                InterlockedAdd(SharedCandidateCount,1,localCandidate);
            }
        }
        // Invalid input still contributes an invalid statistic, but must not
        // address a scalar element outside the bound output view.
        if (tid.x < areaCount) TriangleArea[tid.x]=area;
    }
    SharedStats[lane]=stats;
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0 && SharedCandidateCount != 0 && counterBytes >= 8)
    {
        CandidateCounters.InterlockedAdd(0,SharedCandidateCount,SharedCandidateBase);
        const uint available = SharedCandidateBase < capacity ? capacity-SharedCandidateBase : 0;
        const uint omitted = SharedCandidateCount-min(SharedCandidateCount,available);
        if (omitted != 0) { uint ignored; CandidateCounters.InterlockedAdd(4,omitted,ignored); }
    }
    GroupMemoryBarrierWithGroupSync();
    if (stats.candidateCount != 0 && SharedCandidateBase < capacity && localCandidate < capacity-SharedCandidateBase)
        Candidates[SharedCandidateBase+localCandidate]=tid.x;
    for (uint stride=128; stride>0; stride>>=1)
    {
        if (lane < stride) SharedStats[lane]=MergeStatistics(SharedStats[lane],SharedStats[lane+stride]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0 && group.x < summaryCount)
    {
        GroupSummaries[group.x].statistics=SharedStats[0];
        for (uint i=0;i<64;++i) GroupSummaries[group.x].histogram[i]=SharedHistogram[i];
    }
}
