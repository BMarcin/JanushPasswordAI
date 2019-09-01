from JanushPasswordGAN.CharacterEmbeddings import CharacterEmbeddings as CE
import logging
import random

logging.basicConfig(format='"%(asctime)s    %(message)s', level=logging.INFO)

f = open("JanushPasswordGAN/100k.txt", encoding="UTF8")

hasla = [line.replace("\n", "") for number, line in enumerate(f.readlines()) if len(line.replace("\n", "")) >= 3]

# haselka = []
# while len(haselka) < 100000:
#     los = random.randint(0, len(hasla)-1)
#
#     if hasla[los] not in haselka:
#         haselka.append(hasla[los])
#
# print(len(haselka))
#
# f2 = open("100k.txt", "a")
#
# out = [f2.write(linijka+"\n") for linijka in haselka]
#
# f2.close()
f.close()



# hasla = ["abc", "aaad", "bbbe", "cccff", "abbh", "acc", "acb", "bac"]

ce = CE(hasla, learningrate=1e-4, epochs=10000, hiddenvectors=400, device="auto", printstep=50, batchsize=40000)


