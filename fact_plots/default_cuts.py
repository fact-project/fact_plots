cuts = dict()

cuts["uncutted"] = [
    "Size > 0"
    ]

cuts["qualitycuts"] = [
        "Size > 65",
        "Leakage < 0.5",
        "Leakage2 < 0.85",
        "numIslands < 8",
        "Width < 45",
        "Length < 60",
        # "Concentration_onePixel > 0.01",
        # "Concentration_twoPixel > 0.02",
        "ConcCore > 0",
        # "concCOG > 0.02"
]

cuts["precuts"] = [
        # "Size > 60",
        "Leakage < 0.09",
        "Leakage2 < 0.15",
        "numIslands < 4",
        "Width < 13.25",
        "Length < 37",
        "numPixelInShower > 10",
        "Concentration_onePixel > 0.0325",
        "Concentration_twoPixel > 0.063",
        # "ConcCore > 0",
        "concCOG > 0.055",
]

cuts["analysis4_pre"] = [
    "Leakage < 0.6",
    "Leakage2 < 0.85",
    "numIslands < 8",
    "numPixelInShower >= 10",
    "Width < 35",
    "Length < 70",
]

cuts["ICRC2015_pre_Xtalk"] = [
        "Size > 60",
         "Leakage <= 0.09",
         "Leakage2 <= 0.15",
         "numIslands <2",
         "Width < 10",
         "Length < 28",
        #  "Width*Length*3.141/pow(log10(Size),2) < 80", #area vs. size cut
        "Concentration_onePixel > 0.044",
        "Concentration_twoPixel > 0.09",
]
