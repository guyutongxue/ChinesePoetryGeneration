from plan import Planner
from generator import Generator
from typing import Set, List
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

# For testing purpose.
if __name__ == '__main__':
    planner = Planner()
    generator = Generator()
    while True:
        hint : str = input("Type first sentence: ")
        words : tuple = (hint[0:2], hint[2:4], hint[4:6], hint[6:7])
        hint = ' '.join(words)
        print(hint)
        keywords : List[str] = planner.plan(hint)
        print("Keywords: ", keywords)
        poem : List[str] = generator.generate_chen(keywords, hint)
        output = '\n'.join(poem).replace(' ','')
        print("Poem: \n", output)
        with open("result.txt", 'a', encoding='utf-8') as f:
            f.write("Input: " + hint + '\n')
            f.write("Keywords: " + str(keywords) + '\n')
            f.write("Poem: \n")
            f.write(str(output))
            f.write('\n')
