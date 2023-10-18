from PIL import Image

def make_transparent(mask):
    # Získejte data z masky
    mask_data = mask.getdata()

    # Vytvořte nový obrázek s upravenými daty (nastavte bílou barvu na průhlednou)
    new_mask_data = [(r, g, b, 255) if r == 255 and g == 255 and b == 255 else (r, g, b, 0) for r, g, b in mask_data]

    # Vytvořte novou masku s upravenými daty
    new_mask = Image.new("RGBA", mask.size)
    new_mask.putdata(new_mask_data)
    new_mask.save("generovana_maska.png")
    # Ujistěte se, že maska a obrázek mají stejnou velikost
    new_mask = new_mask.resize(image.size, Image.LANCZOS)
    return new_mask



# Otevřete masku a cílový obrázek
mask = Image.open("mask_paintnet.png")
image = Image.open("230926dr.jpg")


# Aplikujte masku na obrázek
result = Image.new("RGBA", image.size)
result.paste(image.convert("RGBA"), (0,0))
result.paste((255,255,255), (0, 0), mask)

# Uložte změněný obrázek ve formátu PNG
result.save("obrazek_upraveny3.png")
