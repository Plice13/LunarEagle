from PIL import Image

# Otevřete masku a cílový obrázek
mask = Image.open("maska_true_bg.png")
image = Image.open("230926dr.jpg")

# Ujistěte se, že maska a obrázek mají stejnou velikost
mask = mask.resize(image.size, Image.LANCZOS)

# Aplikujte masku na obrázek
image.paste((255, 255, 255), (0, 0), mask)

# Uložte změněný obrázek
image.save("obrazek_upraveny.jpg")
