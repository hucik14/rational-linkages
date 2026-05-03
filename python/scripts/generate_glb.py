import build123d as bd

# to generate glb files for docs visualization
part = bd.import_step("mechanism.step")
part.color = bd.Color(1, 0.5, 0)
bd.export_gltf(part, "mechanism.glb", binary=True)

# test locally with:
# python -m http.server 8000 --directory build/html