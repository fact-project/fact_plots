cuts = dict()

cuts["uncutted"] = [
    "size > 0"
    ]

cuts["qualitycuts"] = [
        "size > 65",
        "leakage1 < 0.5",
        "leakage2 < 0.85",
        "num_islands < 8",
        "width < 45",
        "length < 60",
        # "concentration_one_pixel > 0.01",
        # "concentration_two_pixel > 0.02",
        "concentration_core > 0",
        # "concentration_cog > 0.02"
]

cuts["precuts"] = [
        # "size > 60",
        "leakage1 < 0.09",
        "leakage2 < 0.15",
        "num_islands < 4",
        "width < 13.25",
        "length < 37",
        "num_pixel_in_shower > 10",
        "concentration_one_pixel > 0.0325",
        "concentration_two_pixel > 0.063",
        # "concentration_core > 0",
        "concentration_cog > 0.055",
]

cuts["analysis4_pre"] = [
    "leakage1 < 0.6",
    "leakage2 < 0.85",
    # "num_islands < 8",
    "num_pixel_in_shower >= 10",
    "width < 35",
    "length < 70",
]

cuts["ICRC2015_pre_Xtalk"] = [
        "size > 60",
         "leakage1 <= 0.09",
         "leakage2 <= 0.15",
         "num_islands <2",
         "width < 10",
         "length < 28",
        #  "width*length*3.141/pow(log10(size),2) < 80", #area vs. size cut
        "concentration_one_pixel > 0.044",
        "concentration_two_pixel > 0.09",
]
