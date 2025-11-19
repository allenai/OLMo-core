from matplotlib import font_manager

# Example: look for all installed fonts containing "Manrope"
fonts = font_manager.findSystemFonts(fontpaths=None, fontext="ttf")

# Filter to the family you're interested in
target = "Manrope"
matching_fonts = [f for f in fonts if target.lower() in f.lower()]

for path in matching_fonts:
    prop = font_manager.FontProperties(fname=path)
    name = prop.get_name()
    print(f"{name} -> {path}")
