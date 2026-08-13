#!/usr/bin/env ruby
# frozen_string_literal: true

source = File.read(ARGV.fetch(0))
common = File.read(ARGV.fetch(1))
%w[PsxTextureTriangleSpan PsxTextureSpanSample
   Draw_FillTexturedQuadScanlineGaps].each do |name|
  abort "textured FT4 raster compatibility helper #{name} is missing" unless
    source.include?(name)
end
abort "textured correction does not use the PS1 fixed-point UV integer part" unless
  source.include?("fixed_u >> 16") && source.include?("fixed_v >> 16")
abort "textured correction scans polygon interiors instead of span endpoints" unless
  source.include?("span0.x_start, span0.x_end") &&
  source.include?("span1.x_start, span1.x_end")
abort "obsolete axis-aligned geometry strip is still active" if
  source.include?("uStep") || source.include?("Right endpoint:")

trace = source.index("TraceGpuPrimitive(vertex_cur, nVertices")
correction = source.index("Draw_FillTexturedQuadScanlineGaps(compatibility_quad)")
abort "primitive trace sees synthetic compatibility pixels" unless
  trace && correction && trace < correction

puts "textured FT4 scanline endpoints preserve PS1 UVs without stretching geometry"

abort "pixel tracer does not expose exact triangle-edge membership" unless
  common.include?("TraceTriangleEdges") && common.include?('edge=%s%s%s values=%ld,%ld,%ld')
