import glob

def load_spells_list():
    spells_list = []
    for fil in glob.glob("spells/*.txt"):
        with open(fil) as spell_file:
            spells_list.append(spell_file.read())
    return spells_list