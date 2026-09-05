#!/usr/bin/env ruby
# frozen_string_literal: true

source = File.read(ARGV.fetch(0))
common = File.read(ARGV.fetch(1))
span_source = File.read(File.join(File.dirname(ARGV.fetch(0)), "texture_span.h"))
%w[PsxTextureTriangleSpan PsxTextureSpanSample ModernTextureSample
   Draw_FillTexturedQuadScanlineGaps].each do |name|
  abort "textured FT4 raster compatibility helper #{name} is missing" unless
    source.include?(name)
end
abort "textured correction does not use the PS1 fixed-point UV integer part" unless
  span_source.include?("fixed_u >> 16") && span_source.include?("fixed_v >> 16")
abort "textured correction does not reproduce truncated 16.16 UV steps" unless
  span_source.include?("step_u") && span_source.include?("step_v") &&
  source.include?("TextureSamplesDiffer")
abort "textured correction ignores unstable integral float UV boundaries" unless
  File.read(File.join(File.dirname(ARGV.fetch(0)), "texture_sample.h"))
      .include?("fabs(raw_u - round(raw_u))") &&
  source.include?("bool unstable = ModernTextureSample")
abort "textured Gouraud coverage does not carry PS1 scanline RGB" unless
  source.include?("expected_r") && span_source.include?("step_r") &&
  source.include?("Draw_FillTexturedQuadScanlineGaps(compatibility_quad,")
abort "obsolete axis-aligned geometry strip is still active" if
  source.include?("uStep") || source.include?("Right endpoint:")

trace = source.index("TraceGpuPrimitive(vertex_cur, nVertices")
correction = source.index("Draw_FillTexturedQuadScanlineGaps(compatibility_quad,")
abort "primitive trace sees synthetic compatibility pixels" unless
  trace && correction && trace < correction

puts "textured FT4 scanline endpoints preserve PS1 UVs without stretching geometry"

abort "pixel tracer does not expose exact triangle-edge membership" unless
  common.include?("TraceTriangleEdges") && common.include?('edge=%s%s%s values=%ld,%ld,%ld')
