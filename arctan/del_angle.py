xs=[-359,-270,-181,-180,-179,-90,-1,0,-1,90,179,180,181,270,359,360]
for x in xs:
    print(f'Rozdíl {x} dá po přičtení {x+180}. Po % získáme {(x+180)%360} a po odečtení dostaneme {((x+180)%360)-180}')