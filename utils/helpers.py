from PIL import Image


def add_bg_to_rgba(
    input_path: str,
    output_path: str,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Composite an RGBA PNG onto a solid background color and save as RGB PNG."""
    img = Image.open(input_path).convert("RGBA")
    bg = Image.new("RGBA", img.size, (*color, 255))
    Image.alpha_composite(bg, img).convert("RGB").save(output_path)
