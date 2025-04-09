class Morphology:
    def __init__(self):
        self.words = {
            "help": "helpful",
            "happy": "unhappy",
            "read": "reading",
            "nation": "national",
            "child": "children",
            "act": "actor",
            "run": "runner",
            "govern": "government",
            "write": "rewrite"
        }
    
    def add_affix(self, word):
        return self.words[word] if word in self.words else "No affixation rule found."
    
    def delete_affix(self, affixed_word):
        for base, modified in self.words.items():
            if affixed_word == modified:
                return base
        return "No matching base word found."
    
    def display_table(self):
        print("| Base Word | Addition (Affixation) | New Word | Deletion (Clipping) | Base Word |")
        print("|-----------|-----------------------|----------|----------------------|-----------|")
        for base, modified in self.words.items():
            affix = modified.replace(base, "")
            print(f"| {base:<9} | +{affix:<21} | {modified:<8} | -{affix:<20} | {base:<9} |")

if __name__ == "__main__":
    morph = Morphology()
    morph.display_table()
    
    # Interactive Example
    word = input("Enter a base word to add affix: ").strip().lower()
    print("New word after affixation:", morph.add_affix(word))
    
    affixed_word = input("Enter a word with an affix to remove it: ").strip().lower()
    print("Root word after affix removal:", morph.delete_affix(affixed_word))
