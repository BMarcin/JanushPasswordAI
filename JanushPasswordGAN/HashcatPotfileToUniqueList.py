class HashcatPotfileToUniqueList:
    def __init__(self, path):
        self.__file = open(path)

        self.__hashes = []
        self.__passwords = []

        for linijka in self.__file.readlines():
            linijka = linijka.replace("\n", "").split(":")

            if linijka[0] not in self.__hashes:
                self.__hashes.append(linijka[0])

            if linijka[1] not in self.__passwords:
                self.__passwords.append(linijka[1])

        self.__file.close()

    def get_hashes(self):
        return self.__hashes

    def get_passwords(self):
        return self.__passwords

    def save_passwords_to_file(self, filename):
        file = open(filename, "a")
        for haslo in self.__passwords:
            file.write(haslo+"\n")
        file.close()

    def save_hashes_to_file(self, filename):
        file = open(filename, "a")
        for hash in self.__hashes:
            file.write(hash+"\n")
        file.close()

if __name__ == '__main__':
    hp = HashcatPotfileToUniqueList("hashcat.potfile")
    #print(hp.get_hashes())
    #print(hp.get_passwords())
    hp.save_passwords_to_file("outed.txt")