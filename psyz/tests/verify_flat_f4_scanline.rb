#!/usr/bin/env ruby
# frozen_string_literal: true

source = File.read(ARGV.fetch(0))

%w[PsxFlatTriangleSpan ModernTriangleContains Draw_FillFlatQuadScanlineGaps].each do |name|
  abort "missing flat F4 raster compatibility helper #{name}" unless source.include?(name)
end
abort "F4 correction is not restricted to opaque flat quads" unless
  source.include?("!isGouraud && nVertices == 4") &&
  source.include?("!(code & SEMITRANSP)")
abort "flat F4 correction can redraw already-covered Metal pixels" unless
  source.include?("if (expected && !covered)")
abort "flat F4 correction does not preserve primitive order" unless
  source.index("Draw_EnqueueBuffer(nVertices, nIndices)") <
  source.index("Draw_FillFlatQuadScanlineGaps(compatibility_quad)")

# Rage Racer's real tachometer F4.  Its second triangle produces an inclusive
# PS1 scanline edge which Metal's top-left rule rejects.  Keep this packet as a
# compact regression fixture without adding a game-specific renderer branch.
vertices = [[258, 196], [285, 173], [261, 199], [286, 174]]

def span(points, y)
  v = points.each_with_index.sort_by { |(point, index)| [point[1], index] }.map(&:first)
  return nil if v[2][1] == v[0][1] || y < v[0][1] || y > v[2][1]

  if y < v[1][1]
    return nil if v[1][1] == v[0][1]
    t = (y - v[0][1]).to_f / (v[1][1] - v[0][1])
    x1 = v[0][0] + (v[1][0] - v[0][0]) * t
  else
    return nil if v[2][1] == v[1][1]
    t = (y - v[1][1]).to_f / (v[2][1] - v[1][1])
    x1 = v[1][0] + (v[2][0] - v[1][0]) * t
  end
  t = (y - v[0][1]).to_f / (v[2][1] - v[0][1])
  x2 = v[0][0] + (v[2][0] - v[0][0]) * t
  [x1, x2].min.ceil..[x1, x2].max.floor
end

triangles = [vertices.values_at(0, 1, 2), vertices.values_at(1, 2, 3)]
psx_pixels = triangles.flat_map do |triangle|
  (triangle.map(&:last).min..triangle.map(&:last).max).flat_map do |y|
    range = span(triangle, y)
    range ? range.map { |x| [x, y] } : []
  end
end.uniq

edge_pixels = (174..199).map { |y| [460 - y, y] }
missing = edge_pixels & psx_pixels
abort "tachometer fixture no longer exposes the 26-pixel scanline edge" unless
  missing == edge_pixels

puts "flat F4 scanline compatibility covers the tachometer edge generically"
