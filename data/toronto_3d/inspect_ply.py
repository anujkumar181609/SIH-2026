from plyfile import PlyData

path = "L001.ply"  # adjust filename if different

ply = PlyData.read(path)

print("Elements in file:")
for element in ply.elements:
    print(f"  '{element.name}' — {element.count} entries")
    print(f"  Properties: {[p.name for p in element.properties]}")
    print()

# Show first 3 rows of the main point data
vertex = ply['vertex']
print("First 3 points, all fields:")
for i in range(3):
    print({name: vertex[name][i] for name in vertex.data.dtype.names})