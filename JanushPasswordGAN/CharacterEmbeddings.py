import multiprocessing

import numpy as np

import torch
import torch.nn as nn
from torch import optim

import matplotlib.pyplot as plt

import logging

from joblib import Parallel, delayed


class CBOW(nn.Module):
    def __init__(self, inputs, outputs, hiddenvectors):
        """
        Zbudowanie modelu sieci. Klasa dziedziczy po torch.nn.Module.

        :param hiddenvectors: Ilość neuronow warstwy ukrytej - odpowiada ilości wektorów wyznaczonych dla danego znaku
        :type hiddenvectors: int

        :param inputs: liczba wejść do sieci - wielkość okna
        :type inputs: int

        :param outputs: liczba wyjść z sieci - wielkość jednego one hot vectora
        :type outputs: int
        """

        super(CBOW, self).__init__()

        self.fc_in = nn.Linear(in_features=inputs, out_features=hiddenvectors)
        self.fc_out = nn.Linear(in_features=hiddenvectors, out_features=outputs)

    def forward(self, x):
        """
        Funkcja opisująca przeływanie tensorów przez sieć.

        :param x: Tensor wejściowy. Jego wymiary muszą być zgodne z ilością wejść sieci/
        :type x: torch.Tensor

        :return:
        """

        x = self.fc_in(x)
        x = self.fc_out(x)

        return x


class CharacterEmbeddings:
    def __init__(self, hasla, windowsize=-1, precharembedding_fill=0, learningrate=1e-4, epochs=20000, device='auto', hiddenvectors=4, printstep=1000, multicore = True, batchsize=None, printepochstep=1000):
        logging.info("Witaj w Character Embeddings")
        self._hasla = hasla
        self._batchsize = batchsize
        self._printepochstep = printepochstep

        logging.info("CE: Wyszukiwanie unikalnych znaków...")
        self.__find_unique()
        logging.debug("CE: Unikalne znaki")
        logging.debug(self._unique)

        logging.info("CE: Budowanie one hot vectorów dla każego znaku...")
        self.__make_one_hot_vectors()
        logging.debug("CE: Wyznaczone one hot vectory")
        logging.debug(self._one_hot_vectors)

        logging.info("CE: Tłumaczenie słów na one hot vectory...")
        self.__make_word_vectors()
        logging.debug("CE: Przetłumaczone słowa")
        logging.debug(self._word_vectors)

        logging.info("CE: Wyznaczanie ID klas...")
        self._class_ids = self.__make_class_id()
        logging.debug("CE: Wyznaczone klasy")
        logging.debug(self._class_ids)

        logging.info("CE: Wyszukiwanie najdłuższego słowa...")
        self.__find_longest_word()
        logging.debug("CE: Najdłuższe słowo")
        logging.debug(self.__find_longest_word())

        logging.info("CE: Obliczanie minimalnej wielkości okna...")
        self.__get_min_window_size()
        logging.debug("CE: Minimalna wielkość okna")
        logging.debug(self.__get_min_window_size())

        logging.info("CE: Tworzenie wejść i wyjść dla sieci...")
        if windowsize == -1:
            logging.debug("CE: Pre char embedding: wykorzystywanie minimalnej wielkości okna")
            if multicore:
                self.__make_pre_char_embedding_multiprocessing(self._min_window_size, fill=precharembedding_fill)
            else:
                self.__make_pre_char_embedding(self._min_window_size, fill=precharembedding_fill)
        else:
            logging.debug("CE: Pre char embedding: wykorzystywanie ustalonej wielkości okna")
            if multicore:
                self.__make_pre_char_embedding_multiprocessing(windowsize, fill=precharembedding_fill)
            else:
                self.__make_pre_char_embedding(windowsize, fill=precharembedding_fill)

        logging.debug("CE: Pre char embeddings inputs")
        logging.debug(self._pre_char_embedding_inputs)
        logging.debug("CE: Pre char embeddings outputs")
        logging.debug(self._pre_char_embedding_outputs)

        logging.info("CE: Wybieranie urządzenia...")
        if device == "auto":
            logging.debug("CE: Urządzenie wybierane automatycznie...")
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logging.debug("CE: Wybrane urządzenie")
            logging.debug(self._device)
        else:
            logging.debug("CE: Urządzenie wybierane zgodnie z argumentem...")
            self._device = torch.device(device)

        if device == "cuda:0":
            torch.cuda.empty_cache()
            torch.cuda.init()

        self.__ucz(epochs, learningrate, hiddenvectors, printstep, multicore=multicore)

    def ucz(self, epochs, learningrate, hiddenvectors, printstep, multicore=True):
        self.__ucz(epochs, learningrate, hiddenvectors, printstep, multicore=multicore)

    def __ucz(self, epochs, learningrate, hiddenvectors, printstep, multicore=True):
        logging.info("CE: Witaj w CE - NN...")

        logging.info("CE - NN: Zmiana wymiarów dla wektorów wejściowych...")
        self._pre_char_embedding_inputs = self._pre_char_embedding_inputs.reshape(
            self._pre_char_embedding_inputs.shape[0], -1)

        logging.info("CE - NN: Tłumaczenie wartości wyjściowych na klasy...")

        ''' Wykorzystanie wielowątkowości CPU '''
        if multicore:
            self._training_classes = self.__translate_labels_to_classes_multithreading(self._pre_char_embedding_outputs)
        else:
            self._training_classes = self.__translate_labels_to_classes(self._pre_char_embedding_outputs)

        logging.debug("CE - NN: Wartości wejścia")
        logging.debug(self._pre_char_embedding_inputs)
        logging.debug("CE - NN: Wartości wyjść")
        logging.debug(self._pre_char_embedding_outputs)
        
        logging.info("CE - NN: Posiadam " + str(len(self._pre_char_embedding_inputs)) + " próbek wejściowych")

        logging.info("CE - NN: Budowanie modelu CBOW...")
        logging.info("CE - NN: Ilość wejść: "+ str(self._pre_char_embedding_inputs.shape[1]))
        logging.info("CE - NN: Ilość wyjść " + str(len(self._unique)))
        self._cbow = nn.DataParallel(CBOW(self._pre_char_embedding_inputs.shape[1], len(self._unique), hiddenvectors))
        logging.debug("CE - NN: Model CBOW...")
        logging.debug(self._cbow)

        ''' Scenariusz z obsługą batch'a '''
        if self._batchsize is None:
            ''' Nie korzystamy z batcha, więc ładujemy całość danych do urządzenia docelowego '''
            logging.info("CE - NN: Budowanie tensora wartości wejściowych...")
            self._TensorX = torch.tensor(self._pre_char_embedding_inputs, dtype=torch.float,).to(self._device)
            logging.debug("CE - NN: TensorX")
            logging.debug(self._TensorX)

            logging.info("CE - NN: Budowanie tensora z klasami...")
            self._TensorClass = torch.tensor(self._training_classes).long().to(self._device)
            logging.debug("CE - NN: Tensor klas")
            logging.debug(self._TensorClass)
        else:
            ''' Korzystamy z batcha, więc dzielimy nasze wartości pomiędzy batche i zapisujemy do tablicy
                Taki tensor jest wysyłany do odpowiedniego urządzenia dopiero w momencie, gdy musi przejść
                przez sieć.
            '''
            logging.info("CE - NN: Tworzenie batchy...")
            self._TensorsX = []
            self._TensorsClass = []

            reszta = len(self._pre_char_embedding_inputs) % self._batchsize
            lastx = 0

            for x in range(int(len(self._pre_char_embedding_inputs) / self._batchsize)):
                self._TensorsX.append(torch.tensor(
                    self._pre_char_embedding_inputs[x * self._batchsize:x * self._batchsize + self._batchsize],
                    dtype=torch.float, requires_grad=False))

                self._TensorsClass.append(torch.tensor(
                    self._training_classes[x * self._batchsize:x * self._batchsize + self._batchsize],
                    requires_grad=False).long())

                lastx = x

            if reszta != 0:
                self._TensorsX.append(torch.tensor(
                    self._pre_char_embedding_inputs[lastx * self._batchsize + self._batchsize:], dtype=torch.float))

                self._TensorsClass.append(torch.tensor(self._training_classes[lastx * self._batchsize + self._batchsize:]).long())
            
        logging.info("CE - NN: Przygotowywanie optymizera Adam z learning rate " + str(learningrate) + "...")
        self._optimizer = optim.Adam(self._cbow.parameters(), lr=learningrate, betas= (0.9, 0.99))

        logging.info("CE - NN: Przygotowywanie CrossEntropyLoss...")
        self._criterion = nn.CrossEntropyLoss().to(self._device)

        logging.info("CE - NN: Witaj w trybie uczenia...")
        self._lossx = []
        self._lossy = []

        logging.info("CE - NN: Rozpoczynam naukę...")

        ''' Zapisujemy loss '''
        loss = 0

        ''' Rozpoczynamy epoki '''
        for epoch in range(epochs):
            if self._batchsize is not None:
                ''' Gdy wykorzystujemy batche najpierw musimy przesłać zawartość jednego do odpowiedniego urządzenia '''
                for id in range(len(self._TensorsX)):
                    self._TensorX = self._TensorsX[id].to(self._device)
                    self._TensorClass = self._TensorsClass[id].to(self._device)

                    loss = self.__epoch()
            else:
                ''' Gdy nie korzystamy z batchy po prostu przechodzimy przez sieć '''
                loss = self.__epoch()

            if epoch % printstep == 0:
                logging.info("CE - NN: epoch: " + str(epoch) + " loss: " + str(loss))

        ''' wypisanie końcowego LOSS '''
        logging.info("CE - NN: Końcowy loss: "+str(loss))

        ''' przestawienie sieci na tryb zwykły, a nie uczenia '''
        logging.info("CE - NN: Przestawianie sieci w tryb zwykły...")
        self._cbow.eval()

    def __epoch(self):
        ''' Przestawienie obiektu sieci w tryb uczenia '''
        self._cbow.train()
        self._optimizer.zero_grad()

        ''' Przeliczenie wartości '''
        y_ = self._cbow(self._TensorX).to(self._device)

        ''' Obliczenie funkcji loss '''
        loss = self._criterion(y_, self._TensorClass).to(self._device)

        ''' Dodanie loss do wykresiku '''
        self._lossy.append(loss)

        ''' Wsteczna propagacja i aktualizacja parametrów przez optimizer'''
        loss.backward(loss)
        self._optimizer.step()

        return loss


    def __find_unique(self):
        """
        W hasłach jakie podaliśmy w konstruktorze wyszukujemy unikalne znaki i zwracamy array numpy.

        Dla self._hasla = ["abc", "acde"]
        Otrzymamy: ["a", "b", "c", "d", "e"]

        :return: array unikalnych znaków
        :rtype: numpy.array
        """
        self._unique = np.array(sorted(list({letter for word in self._hasla for letter in word})))
        return self._unique

    def __make_one_hot_vectors(self):
        """
        Dla unikalnych znaków wyznaczamy ich one hot vectory.

        Dla self._uniqe = ["a", "b", "c"]
        Otrzymamy: {
            "a": [1, 0 ,0],
            "b": [0, 1, 0],
            "c": [0, 0, 1]
        }

        :return: słownik, w którym kluczem jest znak, a wartością one hot vector dla tego znaku
        :rtype: dict[string, list[int]]
        """
        self._one_hot_vectors = {self._unique[cnt]: [1 if ii == cnt else 0 for ii in range(len(self._unique))] for cnt
                                 in range(len(self._unique))}
        return self._one_hot_vectors

    def __make_class_id(self):
        """
        Dla każdego one hot vectora (w sumie dla każdego unikalnego znaku) wyznaczamy ID klasy.

        Dla self._one_hot_vectors: {"a": [1, 0], "b": [0, 1]}
        Otrzymamy: {"a": 0, "b": 1}

        :return: id klas dla poszczególnych znaków
        :rtype: dict[char, int]
        """
        self._class_ids = {}

        for cnt, one_hot_vector in enumerate(self._one_hot_vectors):
            self._class_ids[one_hot_vector] = cnt

        return self._class_ids

    def __make_word_vector(self, word):
        """
        Pojedyńcze słowo tłumaczymy na wektor one hot vectorów.

        Dla word = "abc"
        Otrzymamy: [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]

        :param word:
        :type word:str

        :return: słowo przetłumaczone na one hot vector
        :rtype: numpy.array
        """
        return np.array([self._one_hot_vectors[letter] for letter in word])

    def __make_word_vectors(self):
        """
        Dla każdego hasła tworzymy wektor one hot vectorów.

        Dla self._hasla = ["ac", "abc"]
        Otrzymamy: [
            [
                [1, 0, 0],
                [0, 0, 1]
            ],
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]
        ]

        :return: Przetłumaczone słowa na one hot vectory
        :rtype: list[list[list[int]]
        """
        self._word_vectors = [self.__make_word_vector(slowo) for slowo in self._hasla]
        return self._word_vectors

    def __make_pre_char_embedding(self, windowsize, fill=0, returnOnly = False, wektory=None):
        """
        Dla podanych haseł wyznaczyć wszystkie kombinacje wejściowych i wyjściowych one hot vectorów dla znaków zgodnie
        ze schematem CBOW.

        Przykład dla windowsize=4 i hasła "abc" (poszczególne znaki oraz null występują jako one hot vectory)
        One hot vectorem dla null jest wektor składający się z samych zer

        a-----|            null--|            null--|
        b-----|            a-----|            null--|
              |----c             |----b             |----a
        null--|            c-----|            b-----|
        null--|            null--|            c-----|


        Dla windowsize=4 i hasła "abc"
        Otrzymamy:
            inputs = [
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ],
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 0, 0]
                ],
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ]
            ]

            outputs = [
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0]
            ]


        :param windowsize: Wielkość okna w obrębie której należy wyznaczyć wartości wejściowe i wyjściowe. Parametr ten
        jest konieczny, gdyż dla różnych wielkości słów wejściowych trzeba wyznaczyć maksymalną wielkość okna dla
        najdłuższego słowa, by móc ustalić konkretną wielkość wejść i wyjść sieci neuronowej.
        :type windowsize: int

        :param fill: Parametr opcjonalny. Zawiera informacje czym wypełnić puste wektory (wektor null). Parametru w
        sumie nie powinno tu być, gdyż wartość inna niż 0 będzie powodować błędne przeliczenia sieci.
        :type fill: int

        :param returnOnly: Parametr definiuje, czy funkja ma tylko zwrócić wartości, czy też je zapisać do zmiennej
        klasy
        :type returnOnly: bool

        :param wektory: One-hot vectory, po których funkcja ma iterować
        :type wektory: list

        :return: wejścia i wyjścia dla sieci
        :rtype: numpy.array, numpy.array
        """

        ''' definicje wejsc i wyjsc '''
        inns = []
        outs = []

        ''' zapisanie wielkości wektora do pola klasy '''
        if returnOnly == False:
            self._window_size = windowsize

        ''' lecimy po kazdym wektorze one hot vectorow '''
        if returnOnly == True and wektory == None:
            wektory = self._word_vectors

        for wektor in wektory:
            '''dla kazdego wektora sprawdzamy minimalna dlugosc okna, by sprawdzic czy jest zgodnosc z argumentem 
            "windowsize" '''
            min_window_size = self.__get_min_window_size(wektor)

            ''' sprawdzenie poprawności podanych danych '''
            if min_window_size > windowsize:
                raise Exception(
                    "Bledna wielkosc okna, powinna wynosic minimum: " + str(min_window_size) + ", a mam: " + str(
                        windowsize))

            if len(wektor) < 3:
                raise Exception("Wektor zbyt krotki. Dlugosc powinna wynosic minimum 3")

            ''' lokalne wartości inputs i outputs
            w nich  przechowujemy "kombinacje" dla wszystkich znaków jednego słowa'''
            inputs = []
            outputs = []

            ''' tutaj jest bardzo ważna wartość wyliczana 
            mając wielkość okna = 8, a słowo wielkości 3 musimy tak ułożyć znaki tego słowa,
            by znajdowały się one na środkowych pozycjach 
            ta wartość jest indeksem, o który należy przesunąć znaki słowa '''
            beginfill = int((windowsize - min_window_size) / 2)

            ''' outed zawiera index one hot vectora słowa, które znajduje się na wyjściu sieci '''
            outed = len(wektor) - 1

            ''' zbudowanie pierwotnego okna
            początek okna wypełniamy one hot vectorami znaków,
            a resztę zerami '''
            window = [
                wektor[x - beginfill] if (x < len(wektor) + beginfill - 1) and (x > beginfill - 1) else
                [fill for __ in range(len(wektor[0]))] for x in range(windowsize)
            ]

            ''' dodajemy do tablic '''
            inputs.append(window)
            outputs.append(wektor[outed])

            ''' i tutaj główna pętla przekształceń
            dużo rozkmin było jak zrobić wszystkie kombinacje dla danego słowa,
            ale python jest wspaniały jeżeli chodzi o operacje na tablicach,
            więc przesunięcie wszystkich znaków o jeden do przodu to po prostu pierwsza instrukcja pętli 💚
            następnie robimy podmiankę wartości wyjściowej
            i zamieniamy indeks wyjścia '''
            for x in range(len(wektor) - 1):
                window = [window[-1]] + window[:-1]
                window[outed + beginfill + x] = wektor[outed]
                outed = outed - 1

                inputs.append(window)
                outputs.append(wektor[outed])

            ''' inputy i outputy dla konkretnego słowa dodajemy do inputów i outputów całościowych '''
            inns = inns + inputs
            outs = outs + outputs

        if returnOnly == False:
            self._pre_char_embedding_inputs = np.array(inns, dtype='float32')
            self._pre_char_embedding_outputs = np.array(outs, dtype='int')

        return np.array(inns, dtype='float32'), np.array(outs, dtype='int')

    def __make_pre_char_embedding_multiprocessing(self, windowsize, fill=0):
        """
        :param windowsize: Wielkość okna na jakiej ma działać funkcja
        :type windowsize: int

        :param fill: Czym wypełnić pustkę w one-hot-vectorach
        :type fill: int

        :return: wejścia i wyjścia dla sieci
        :rtype: numpy.array, numpy.array
        """
        self._window_size = windowsize

        inns = []
        outs = []

        threads = multiprocessing.cpu_count()

        ''' sprawdzenie poprawności podanych danych '''
        if self.__get_min_window_size() > windowsize:
            raise Exception(
                "Bledna wielkosc okna, powinna wynosic minimum: " + str(self.__get_min_window_size()) + ", a mam: " + str(
                    windowsize))

        vecs = []

        val = 0

        ''' podzielenie funkcji na ilość wątków CPU '''
        for x in range(threads):
            startval = val

            if x < len(self._word_vectors)%threads:
                endval = val+int(len(self._word_vectors)/threads)
                val = val + int(len(self._word_vectors) / threads) + 1
            else:
                endval = val+int(len(self._word_vectors)/threads)-1
                val = val + int(len(self._word_vectors) / threads)

            vecs.append([startval, endval])

        ''' współbieżny for '''
        results = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(self.__pre_char_embedding_process)(vecs[i][0], vecs[i][1], windowsize, fill) for i in range(threads))

        for result in results:
            inns = inns + result[0]
            outs = outs + result[1]

        self._pre_char_embedding_inputs = np.array(inns, dtype='float32')
        self._pre_char_embedding_outputs = np.array(outs, dtype='long')

        return self._pre_char_embedding_inputs, self._pre_char_embedding_outputs

    def __pre_char_embedding_process(self, startindex, outindex, windowsize, fill):
        """
        Funkcja wykonawcza dla współbieżnego for'a

        :param startindex: Indeks od którego pętla ma zaczynać działanie
        :type startindex: int

        :param outindex: Indeks na którym pętla ma zakończyć działanie
        :type outindex: int

        :param windowsize: Wielkość okna na którym ma działać funkcja
        :type windowsize: int

        :param fill: Czym wypełnić pustkę w one-hot vectorach
        :type fill: int

        :return: Zwraca array'e wartości wejściowych i wyjściowych
        :rtype: numpy.array, numpy.array
        """
        inns = []
        outs = []

        for xxx in range(startindex, outindex+1):
            wektor =self._word_vectors[xxx]

            min_window_size = self.__get_min_window_size(wektor)

            if len(wektor) < 3:
                raise Exception("Wektor zbyt krotki. Dlugosc powinna wynosic minimum 3")

            inputs = []
            outputs = []

            beginfill = int((windowsize - min_window_size) / 2)

            outed = len(wektor) - 1

            window = [
                wektor[x - beginfill] if (x < len(wektor) + beginfill - 1) and (x > beginfill - 1) else
                [fill for __ in range(len(wektor[0]))] for x in range(windowsize)
            ]

            inputs.append(window)
            outputs.append(wektor[outed])

            for x in range(len(wektor) - 1):
                window = [window[-1]] + window[:-1]
                window[outed + beginfill + x] = wektor[outed]
                outed = outed - 1

                inputs.append(window)
                outputs.append(wektor[outed])

            inns = inns + inputs
            outs = outs + outputs

        return inns, outs

    def __find_longest_word(self):
        """
        Funkcja pomocnicza.
        Wyszukuje najdłuższe słowo w self._hasla


        :return: Najdłuższe słowo
        :rtype: str
        """
        longest = 0

        for slowo in self._hasla:
            if len(slowo) > longest:
                longest = len(slowo)

        self._longest_word = longest
        return self._longest_word

    def __get_min_window_size(self, word=None):
        """
        Funkcja ma za zadanie znaleźć minimalną wielkość okna.

        Jeżeli parametr "word" nie jest ustawiony na None należy tam podać słowo i to dla niego zostanie wyznaczona
        wartość minimalnej wielkości okna zapewniająca możliwość wykonania wszystkich przekształceń.

        Jeżeli parametr "word" jest ustawiony na None, to wyznaczana jest minimalna wielkość okna dla najdłuższego słowa
        w zbiorze.

        :param word:
        :type word: str|None

        :return: Minimalna wielkość okna dla podanego przypadku
        :rtype: int
        """
        if word is None:
            self._min_window_size = 2 * (self.__find_longest_word() - 1)
            return self._min_window_size
        else:
            return 2 * (len(word) - 1)

    def __translate_class_to_sign(self, classid):
        """
        Funkcja pomocnicza.

        Tłumaczy klasę (wyznaczone wcześniej ID one hot vectora/znaku) na znak

        :param classid: Id klasy, którą chcemy przetłumaczyć
        :type classid: int

        :return: Przypisany znak
        :rtype: str
        """
        for item in self._class_ids:
            if self._class_ids[item] == classid:
                return item

    def __translate_sign_to_class(self, sign):
        """
        Funkcja pomocnicza.

        Tłumaczy znak na wcześniej wyznaczoną klasę.

        :param sign: Znak do przetłumaczenia
        :type sign: str

        :return: Wyznaczona klasa dla znaku
        :rtype: int
        """
        return self._class_ids[sign]

    def __translate_one_hot_vector_to_sign(self, one_hot_vector):
        """
        Funkcja pomocnicza.

        Tłumaczy one hot vector na znak

        :param one_hot_vector:
        :return:
        """
        for item in self._one_hot_vectors:
            if np.array_equal(np.array(self._one_hot_vectors[item]), one_hot_vector):
                return item

    def __translate_labels_to_classes(self, labels):
        """
        Funkcja pomocnicza.

        Tłumaczy wektor labels(wektor one hot vectorów) na klasy. Wektor labels jest wektorem wyjść z funkcji
        make_pre_char_embeddings.

        :param labels: Wektor one hot vectorów.
        :type labels: numpy.array

        :return:  przetłumaczone labele na klasy
        :rtype numpy.array
        """
        return np.array(
            [self.__translate_sign_to_class(self.__translate_one_hot_vector_to_sign(item)) for item in labels])

    def __translate_labels_to_classes_multithreading(self, labels):
        """
        Funkcja tłumaczy labele(one hot vectory) wartośći wejśćiowych na klasy znaków

        :param labels: wyjścia sieci z pre_char_embeddings
        :type labels: numpy.array

        :return: Klasy wyjść do sieci eembeddings
        :rtype: numpy.array
        """
        threads = multiprocessing.cpu_count()

        vecs = []

        val = 0
        for x in range(threads):
            startval = val

            if x < len(labels) % threads:
                endval = val + int(len(labels) / threads)
                val = val + int(len(labels) / threads) + 1
            else:
                endval = val + int(len(labels) / threads) - 1
                val = val + int(len(labels) / threads)

            vecs.append([startval, endval])

        results = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(self.__translate_labels_to_classes_process)(labels[vecs[i][0]:vecs[i][1]+1]) for i in range(threads))

        res = []

        for result in results:
            res = res + result

        return np.array(res, dtype='long').reshape(-1)

    def __translate_labels_to_classes_process(self, labels):
        """
        Funkcja dla współbieznego for'a do tłumaczenia labelsów na klasy

        :param labels: Wyjścia sieci z pre_char_embeddings
        :type labels: numpy.array

        :return: Przetłumaczona lista klas
        :rtype: list
        """
        return [self.__translate_sign_to_class(self.__translate_one_hot_vector_to_sign(item)) for item in labels]

    def test_word(self, word, fill=0):
        """
        WYKORZYSTYWANA EMOTKA: 🈳, WE SE SKOPIUJ

        Funkcja, która ma za zadanie sprawdzić prawdopodobieństwo wystąpienia danego znaku w danej sekwencji po
        nauczeniu sieci.

        Funkcja ta wykorzystuje pewien myk. Nie jest on do końca poprawny, ale działa, a jak działa i jest głupie, to
        w sumie nie jest aż tak głupie.

        Znaki naszego słowa są bezpośrednio konwertowane na one hot vectory. Oznacza to, że musimy wybrać jakiś znak,
        który nie wystąpi w zbiorze znaków. Odwołując się do zbioru naszych haseł jestem przekonany, że na pewno nie
        wystąpi w nim... emotka 🈳 "japoński przycisk 'wolne miejsce'".

        Należy tu również pamiętać o miejscu, w którym wystąpi predykcja. Będzie to znak środkowy w podawanym słowie.
        Np. dla okna = 8 będzie to znak po 4 miejscu w słowie.

        Przykład
        Mamy wielkość okna = 8, chcemy sprawdzić predykcję dla znaku pomiędzy "abb" oraz "cc". Nasze słowo musimy zatem
        zapisać jako: 🈳abbcc🈳🈳.

        :param word: słowo do predykcji litery
        :type word: str

        :param fill: wypełnienie wektora null
        :type fill: int
        """

        ''' utworzenie one hot vectora '''
        logging.info("CE - NN: Witaj w trybie testowym...")
        logging.info("CE - NN: Budowanie one hot vectora dla słowa testowego...")
        word_one_hot = np.array([self._one_hot_vectors[letter]
                                 if letter != '🈳' else [fill for __ in range(len(self._unique))] for letter in word])
        logging.debug("CE - NN: Słowo testowe")
        logging.debug(word_one_hot)

        ''' sprawdzenie zgodności z wielkością okna '''
        if len(word_one_hot) != self._min_window_size:
            raise Exception("Slowo o blednej dluosci wejsciowej")

        ''' zbudowanie tensora '''
        logging.info("CE - NN: Budowanie tensora testowego, zmiana rozmiarów i wysyłanie do odpowiedniego urządzenia...")
        test_tensor = torch.Tensor(word_one_hot.reshape(1, -1)).to(self._device)
        logging.debug("CE - NN: Testowy tensor")
        logging.debug(test_tensor)

        ''' zbudowanie listy tensorów klasowych do podania do CrossEntropyLoss '''
        testing_tensors_class = [torch.tensor([x]).long().to(self._device) for x in range(len(self._unique))]

        ''' robimy tablicę wyników, by móc ją później posortować '''
        prawdopodobienstwa = {}

        ''' wyłączamy gradienty w sieci '''
        logging.info("CE - NN: Wyłączanie gradientów...")
        with torch.no_grad():
            ''' przeliczenie wartości '''
            logging.info("CE - NN: Przeliczanie wartości...")
            y_ = self._cbow(test_tensor)

            logging.info("CE - NN: Obliczanie CrossEntropyLoss dla każdej z klas...")
            for x in range(len(self._unique)):
                ''' obliczenie CrossEntropyLoss i zapisanie do słownika '''
                prawdopodobienstwa[self.__translate_class_to_sign(x)] = self._criterion(y_,
                                                                                        testing_tensors_class[x]).item()
            ''' posortowanie i wypisanie predykcji '''
            logging.info("CE - NN: Sortowanie prawdopodobieństw...")
            for item in sorted(prawdopodobienstwa, key=prawdopodobienstwa.get):
                print(item, prawdopodobienstwa[item])

    def single_similarity(self, letter1, letter2):
        """
        Funkcja do sprawdzania podobieństwa znaków.

        :param letter1: Jeden ze znaków dla których sieć ma już wyznaczony one-hot vector
        :type letter1: str

        :param letter2: Drugi ze znakód dla których sieć ma już wyznaczony one-hot vector

        :return: Podobieństwo znaków
        :rtype: float
        """
        id1 = 0
        id2 = 0

        for classid in self._class_ids:
            if classid == letter1:
                id1 = self._class_ids[classid]

            if classid == letter2:
                id2 = self._class_ids[classid]

        v1 = self._cbow.cpu().module.fc_out.weight[id1].detach().numpy()
        v2 = self._cbow.cpu().module.fc_out.weight[id2].detach().numpy()

        d = np.dot(v1, v2)

        l1 = np.sqrt(np.sum(v1 ** 2))
        l2 = np.sqrt(np.sum(v2 ** 2))

        self._cbow = self._cbow.to(self._device)

        return d / (l1 * l2)

    def print_similarity(self, letter):
        """
        Sprawdzanie wszystkich znanych sieci liter z jedną podaną

        :param letter: Litera, z którą należy sprawdzić prawdopodobieństwo
        :type letter: string
        """
        zbior = {}

        for classid in self._class_ids:
            zbior[classid] =  self.single_similarity(letter, classid)

        for item in sorted(zbior, key=zbior.get, reverse=True):
            print(letter + " oraz " + item + ": ", zbior[item])

    def wykres_czestosci_liter(self):

        self._czestosc = {}

        for haslo in self._hasla:
            for literka in haslo:
                if literka not in self._czestosc:
                    self._czestosc[literka] = 1
                else:
                    self._czestosc[literka] = self._czestosc[literka] + 1

        xs = list(self._czestosc.keys())
        ys = [self._czestosc[ilosc] for ilosc in self._czestosc]

        fix, ax = plt.subplots()
        ax.plot(xs, ys, ".")
        plt.show()

    def jaka_dlugosc_slowa(self):
        """
        Funkcja pomocnicza.

        Ma tylko pokazać wielkość okna, żeby dopasować słowo testujące do ustalonych wartości.
        """
        print(self._window_size)
        print("A tu emotka: 🈳")

    def wykresLoss(self):
        """
        Funkcja ma wygenerować wykres funkcji loss
        """
        self._lossx = [x for x in range(len(self._lossy))]

        fix, ax = plt.subplots()
        ax.plot(self._lossx, self._lossy, ".")
        plt.show()

    def get_character_embeddings(self):
        """ przesłanie wartości wag do cpu i konwersja na numpy """
        self._out_weights = self._cbow.module.fc_out.weight.cpu().detach().numpy()

        self.embeddings = {}

        for cnt, znak in enumerate(self._unique):
            self.embeddings[znak] = self._out_weights[cnt]

            logging.info(" "+znak + ": " + str(self._out_weights[cnt]))

    def check_literowka(self, slowo, marginesbledu=4):
        """
        sprawdzenie poprawności słowa wejściowego

        :param slowo: Słowo do sprawdzenia
        :type slowo: string

        :param marginesbledu: Margines prawdopodobieństwa jaki jest dopuszczalny do rozróżnienia między dobrym, a złym
        słowem
        :type marginesbledu: int
        """
        if len(slowo) < 3:
            raise Exception("Slowo zbyt krotkie. Dlugosc powinna wynosic minimum 3")

        ''' Tworzymy one hot vectory dla słowa '''
        wektory = [self.__make_word_vector(slowo)]

        ''' Tworzymy wartości wejściowe dla sieci '''
        testing_data, _ = self.__make_pre_char_embedding(self._window_size, returnOnly=True, wektory=wektory)

        ''' Tworzymy tablicę z literami, jakie może zawierać słowo w sieci
            [
                [a, b, c], - odnosi się do pierwszej litery słowa sprawdzanego
                [c], - odnosi się do drugiej litery słowa sprawdzanego
                [d, e] - odnosi się do trzeciej litery słowa sprawdzanego
            ]
        '''
        skladanywyraz = [[] for _ in range(len(testing_data))]

        ''' Iterujemy po każdej literce słowa sprawdzajac czy sieć wskaże je jako prawdopodobne w marginesie '''
        for i in range(len(testing_data)):
            ''' wysyłamy tensor do odpowiedniego backendu '''
            test_tensor = torch.Tensor(testing_data[i].reshape(1, -1)).to(self._device)

            ''' Tworzymy tensor klas do sprawdzenia '''
            testing_tensors_class = [torch.tensor([x]).long().to(self._device) for x in range(len(self._unique))]

            ''' robimy tablicę wyników, by móc ją później posortować '''
            prawdopodobienstwa = {}

            ''' wyłączamy gradienty w sieci '''
            with torch.no_grad():
                ''' przeliczenie wartości '''
                y_ = self._cbow(test_tensor)

                for x in range(len(self._unique)):
                    ''' obliczenie CrossEntropyLoss i zapisanie do słownika '''
                    prawdopodobienstwa[self.__translate_class_to_sign(x)] = self._criterion(y_, testing_tensors_class[x]).item()

                ''' posortowanie i wypisanie predykcji '''
                posortowane = []

                for index, item in enumerate(sorted(prawdopodobienstwa, key=prawdopodobienstwa.get)):
                    posortowane.append([item, prawdopodobienstwa[item]])

                    ''' Jezeli nasza predykcja mieści się w marginesie, to dopisujemy ją do tablicy '''
                    if prawdopodobienstwa[item] < marginesbledu:
                        skladanywyraz[len(testing_data) - 1 - i].append(item)

        for wyraz in skladanywyraz:
            print(wyraz)

        ''' Teraz sprawdzamy, gdzie mamy literówkę '''
        for index, literka in enumerate(slowo):
            if literka not in skladanywyraz[index]:
                print("Błędne słowo")
                return

        print("\nSłowo poprawne")
        return

    def save_embeddings_to_file(self, filename):
        f1 = open(filename, "w")
        for xx in ce.embeddings:
            f1.write(xx + ": [")
            for xx2 in ce.embeddings[xx]:
                f1.write(str(xx2) + ", ")
            f1.write("]\n")
        f1.close()


if __name__ == '__main__':
    logging.basicConfig(format='"%(asctime)s    %(message)s', level=logging.INFO)

    hasla = ["abcdefgh", "abcdefhg", "abcdegfh", "abcdeghf", "abcdehfg", "abcdehgf", "abcdfegh", "abcdfehg", "abcdfgeh", "abcdfghe", "abcdfheg", "abcdfhge", "abcdgefh", "abcdgehf", "abcdgfeh", "abcdgfhe", "abcdghef", "abcdghfe", "abcdhefg", "abcdhegf", "abcdhfeg", "abcdhfge", "abcdhgef", "abcdhgfe", "abcedfgh", "abcedfhg", "abcedgfh", "abcedghf", "abcedhfg", "abcedhgf", "abcefdgh", "abcefdhg", "abcefgdh", "abcefghd", "abcefhdg", "abcefhgd", "abcegdfh", "abcegdhf", "abcegfdh", "abcegfhd", "abceghdf", "abceghfd", "abcehdfg", "abcehdgf", "abcehfdg", "abcehfgd", "abcehgdf", "abcehgfd", "abcfdegh", "abcfdehg", "abcfdgeh", "abcfdghe", "abcfdheg", "abcfdhge", "abcfedgh", "abcfedhg", "abcfegdh", "abcfeghd", "abcfehdg", "abcfehgd", "abcfgdeh", "abcfgdhe", "abcfgedh", "abcfgehd", "abcfghde", "abcfghed", "abcfhdeg", "abcfhdge", "abcfhedg", "abcfhegd", "abcfhgde", "abcfhged", "abcgdefh", "abcgdehf", "abcgdfeh", "abcgdfhe", "abcgdhef", "abcgdhfe", "abcgedfh", "abcgedhf", "abcgefdh", "abcgefhd", "abcgehdf", "abcgehfd", "abcgfdeh", "abcgfdhe", "abcgfedh", "abcgfehd", "abcgfhde", "abcgfhed", "abcghdef", "abcghdfe", "abcghedf", "abcghefd", "abcghfde", "abcghfed", "abchdefg", "abchdegf", "abchdfeg", "abchdfge", "abchdgef", "abchdgfe", "abchedfg", "abchedgf", "abchefdg", "abchefgd", "abchegdf", "abchegfd", "abchfdeg", "abchfdge", "abchfedg", "abchfegd", "abchfgde", "abchfged", "abchgdef", "abchgdfe", "abchgedf", "abchgefd", "abchgfde", "abchgfed", "abdcefgh", "abdcefhg", "abdcegfh", "abdceghf", "abdcehfg", "abdcehgf", "abdcfegh", "abdcfehg", "abdcfgeh", "abdcfghe", "abdcfheg", "abdcfhge", "abdcgefh", "abdcgehf", "abdcgfeh", "abdcgfhe", "abdcghef", "abdcghfe", "abdchefg", "abdchegf", "abdchfeg", "abdchfge", "abdchgef", "abdchgfe", "abdecfgh", "abdecfhg", "abdecgfh", "abdecghf", "abdechfg", "abdechgf", "abdefcgh", "abdefchg", "abdefgch", "abdefghc", "abdefhcg", "abdefhgc", "abdegcfh", "abdegchf", "abdegfch", "abdegfhc", "abdeghcf", "abdeghfc", "abdehcfg", "abdehcgf", "abdehfcg", "abdehfgc", "abdehgcf", "abdehgfc", "abdfcegh", "abdfcehg", "abdfcgeh", "abdfcghe", "abdfcheg", "abdfchge", "abdfecgh", "abdfechg", "abdfegch", "abdfeghc", "abdfehcg", "abdfehgc", "abdfgceh", "abdfgche", "abdfgech", "abdfgehc", "abdfghce", "abdfghec", "abdfhceg", "abdfhcge", "abdfhecg", "abdfhegc", "abdfhgce", "abdfhgec", "abdgcefh", "abdgcehf", "abdgcfeh", "abdgcfhe", "abdgchef", "abdgchfe", "abdgecfh", "abdgechf", "abdgefch", "abdgefhc", "abdgehcf", "abdgehfc", "abdgfceh", "abdgfche", "abdgfech", "abdgfehc", "abdgfhce", "abdgfhec", "abdghcef", "abdghcfe", "abdghecf", "abdghefc", "abdghfce", "abdghfec", "abdhcefg", "abdhcegf", "abdhcfeg", "abdhcfge", "abdhcgef", "abdhcgfe", "abdhecfg", "abdhecgf", "abdhefcg", "abdhefgc", "abdhegcf", "abdhegfc", "abdhfceg", "abdhfcge", "abdhfecg", "abdhfegc", "abdhfgce", "abdhfgec", "abdhgcef", "abdhgcfe", "abdhgecf", "abdhgefc", "abdhgfce", "abdhgfec", "abecdfgh", "abecdfhg", "abecdgfh", "abecdghf", "abecdhfg", "abecdhgf", "abecfdgh", "abecfdhg", "abecfgdh", "abecfghd", "abecfhdg", "abecfhgd", "abecgdfh", "abecgdhf", "abecgfdh", "abecgfhd", "abecghdf", "abecghfd", "abechdfg", "abechdgf", "abechfdg", "abechfgd", "abechgdf", "abechgfd", "abedcfgh", "abedcfhg", "abedcgfh", "abedcghf", "abedchfg", "abedchgf", "abedfcgh", "abedfchg", "abedfgch", "abedfghc", "abedfhcg", "abedfhgc", "abedgcfh", "abedgchf", "abedgfch", "abedgfhc", "abedghcf", "abedghfc", "abedhcfg", "abedhcgf", "abedhfcg", "abedhfgc", "abedhgcf", "abedhgfc", "abefcdgh", "abefcdhg", "abefcgdh", "abefcghd", "abefchdg", "abefchgd", "abefdcgh", "abefdchg", "abefdgch", "abefdghc", "abefdhcg", "abefdhgc", "abefgcdh", "abefgchd", "abefgdch", "abefgdhc", "abefghcd", "abefghdc", "abefhcdg", "abefhcgd", "abefhdcg", "abefhdgc", "abefhgcd", "abefhgdc", "abegcdfh", "abegcdhf", "abegcfdh", "abegcfhd", "abegchdf", "abegchfd", "abegdcfh", "abegdchf", "abegdfch", "abegdfhc", "abegdhcf", "abegdhfc", "abegfcdh", "abegfchd", "abegfdch", "abegfdhc", "abegfhcd", "abegfhdc", "abeghcdf", "abeghcfd", "abeghdcf", "abeghdfc", "abeghfcd", "abeghfdc", "abehcdfg", "abehcdgf", "abehcfdg", "abehcfgd", "abehcgdf", "abehcgfd", "abehdcfg", "abehdcgf", "abehdfcg", "abehdfgc", "abehdgcf", "abehdgfc", "abehfcdg", "abehfcgd", "abehfdcg", "abehfdgc", "abehfgcd", "abehfgdc", "abehgcdf", "abehgcfd", "abehgdcf", "abehgdfc", "abehgfcd", "abehgfdc", "abfcdegh", "abfcdehg", "abfcdgeh", "abfcdghe", "abfcdheg", "abfcdhge", "abfcedgh", "abfcedhg", "abfcegdh", "abfceghd", "abfcehdg", "abfcehgd", "abfcgdeh", "abfcgdhe", "abfcgedh", "abfcgehd", "abfcghde", "abfcghed", "abfchdeg", "abfchdge", "abfchedg", "abfchegd", "abfchgde", "abfchged", "abfdcegh", "abfdcehg", "abfdcgeh", "abfdcghe", "abfdcheg", "abfdchge", "abfdecgh", "abfdechg", "abfdegch", "abfdeghc", "abfdehcg", "abfdehgc", "abfdgceh", "abfdgche", "abfdgech", "abfdgehc", "abfdghce", "abfdghec", "abfdhceg", "abfdhcge", "abfdhecg", "abfdhegc", "abfdhgce", "abfdhgec", "abfecdgh", "abfecdhg", "abfecgdh", "abfecghd", "abfechdg", "abfechgd", "abfedcgh", "abfedchg", "abfedgch", "abfedghc", "abfedhcg", "abfedhgc", "abfegcdh", "abfegchd", "abfegdch", "abfegdhc", "abfeghcd", "abfeghdc", "abfehcdg", "abfehcgd", "abfehdcg", "abfehdgc", "abfehgcd", "abfehgdc", "abfgcdeh", "abfgcdhe", "abfgcedh", "abfgcehd", "abfgchde", "abfgched", "abfgdceh", "abfgdche", "abfgdech", "abfgdehc", "abfgdhce", "abfgdhec", "abfgecdh", "abfgechd", "abfgedch", "abfgedhc", "abfgehcd", "abfgehdc", "abfghcde", "abfghced", "abfghdce", "abfghdec", "abfghecd", "abfghedc", "abfhcdeg", "abfhcdge", "abfhcedg", "abfhcegd", "abfhcgde", "abfhcged", "abfhdceg", "abfhdcge", "abfhdecg", "abfhdegc", "abfhdgce", "abfhdgec", "abfhecdg", "abfhecgd", "abfhedcg", "abfhedgc", "abfhegcd", "abfhegdc", "abfhgcde", "abfhgced", "abfhgdce", "abfhgdec", "abfhgecd", "abfhgedc", "abgcdefh", "abgcdehf", "abgcdfeh", "abgcdfhe", "abgcdhef", "abgcdhfe", "abgcedfh", "abgcedhf", "abgcefdh", "abgcefhd", "abgcehdf", "abgcehfd", "abgcfdeh", "abgcfdhe", "abgcfedh", "abgcfehd", "abgcfhde", "abgcfhed", "abgchdef", "abgchdfe", "abgchedf", "abgchefd", "abgchfde", "abgchfed", "abgdcefh", "abgdcehf", "abgdcfeh", "abgdcfhe", "abgdchef", "abgdchfe", "abgdecfh", "abgdechf", "abgdefch", "abgdefhc", "abgdehcf", "abgdehfc", "abgdfceh", "abgdfche", "abgdfech", "abgdfehc", "abgdfhce", "abgdfhec", "abgdhcef", "abgdhcfe", "abgdhecf", "abgdhefc", "abgdhfce", "abgdhfec", "abgecdfh", "abgecdhf", "abgecfdh", "abgecfhd", "abgechdf", "abgechfd", "abgedcfh", "abgedchf", "abgedfch", "abgedfhc", "abgedhcf", "abgedhfc", "abgefcdh", "abgefchd", "abgefdch", "abgefdhc", "abgefhcd", "abgefhdc", "abgehcdf", "abgehcfd", "abgehdcf", "abgehdfc", "abgehfcd", "abgehfdc", "abgfcdeh", "abgfcdhe", "abgfcedh", "abgfcehd", "abgfchde", "abgfched", "abgfdceh", "abgfdche", "abgfdech", "abgfdehc", "abgfdhce", "abgfdhec", "abgfecdh", "abgfechd", "abgfedch", "abgfedhc", "abgfehcd", "abgfehdc", "abgfhcde", "abgfhced", "abgfhdce", "abgfhdec", "abgfhecd", "abgfhedc", "abghcdef", "abghcdfe", "abghcedf", "abghcefd", "abghcfde", "abghcfed", "abghdcef", "abghdcfe", "abghdecf", "abghdefc", "abghdfce", "abghdfec", "abghecdf", "abghecfd", "abghedcf", "abghedfc", "abghefcd", "abghefdc", "abghfcde", "abghfced", "abghfdce", "abghfdec", "abghfecd", "abghfedc", "abhcdefg", "abhcdegf", "abhcdfeg", "abhcdfge", "abhcdgef", "abhcdgfe", "abhcedfg", "abhcedgf", "abhcefdg", "abhcefgd", "abhcegdf", "abhcegfd", "abhcfdeg", "abhcfdge", "abhcfedg", "abhcfegd", "abhcfgde", "abhcfged", "abhcgdef", "abhcgdfe", "abhcgedf", "abhcgefd", "abhcgfde", "abhcgfed", "abhdcefg", "abhdcegf", "abhdcfeg", "abhdcfge", "abhdcgef", "abhdcgfe", "abhdecfg", "abhdecgf", "abhdefcg", "abhdefgc", "abhdegcf", "abhdegfc", "abhdfceg", "abhdfcge", "abhdfecg", "abhdfegc", "abhdfgce", "abhdfgec", "abhdgcef", "abhdgcfe", "abhdgecf", "abhdgefc", "abhdgfce", "abhdgfec", "abhecdfg", "abhecdgf", "abhecfdg", "abhecfgd", "abhecgdf", "abhecgfd", "abhedcfg", "abhedcgf", "abhedfcg", "abhedfgc", "abhedgcf", "abhedgfc", "abhefcdg", "abhefcgd", "abhefdcg", "abhefdgc", "abhefgcd", "abhefgdc", "abhegcdf", "abhegcfd", "abhegdcf", "abhegdfc", "abhegfcd", "abhegfdc", "abhfcdeg", "abhfcdge", "abhfcedg", "abhfcegd", "abhfcgde", "abhfcged", "abhfdceg", "abhfdcge", "abhfdecg", "abhfdegc", "abhfdgce", "abhfdgec", "abhfecdg", "abhfecgd", "abhfedcg", "abhfedgc", "abhfegcd", "abhfegdc", "abhfgcde", "abhfgced", "abhfgdce", "abhfgdec", "abhfgecd", "abhfgedc", "abhgcdef", "abhgcdfe", "abhgcedf", "abhgcefd", "abhgcfde", "abhgcfed", "abhgdcef", "abhgdcfe", "abhgdecf", "abhgdefc", "abhgdfce", "abhgdfec", "abhgecdf", "abhgecfd", "abhgedcf", "abhgedfc", "abhgefcd", "abhgefdc", "abhgfcde", "abhgfced", "abhgfdce", "abhgfdec", "abhgfecd", "abhgfedc", "acbdefgh", "acbdefhg", "acbdegfh", "acbdeghf", "acbdehfg", "acbdehgf", "acbdfegh", "acbdfehg", "acbdfgeh", "acbdfghe", "acbdfheg", "acbdfhge", "acbdgefh", "acbdgehf", "acbdgfeh", "acbdgfhe", "acbdghef", "acbdghfe", "acbdhefg", "acbdhegf", "acbdhfeg", "acbdhfge", "acbdhgef", "acbdhgfe", "acbedfgh", "acbedfhg", "acbedgfh", "acbedghf", "acbedhfg", "acbedhgf", "acbefdgh", "acbefdhg", "acbefgdh", "acbefghd", "acbefhdg", "acbefhgd", "acbegdfh", "acbegdhf", "acbegfdh", "acbegfhd", "acbeghdf", "acbeghfd", "acbehdfg", "acbehdgf", "acbehfdg", "acbehfgd", "acbehgdf", "acbehgfd", "acbfdegh", "acbfdehg", "acbfdgeh", "acbfdghe", "acbfdheg", "acbfdhge", "acbfedgh", "acbfedhg", "acbfegdh", "acbfeghd", "acbfehdg", "acbfehgd", "acbfgdeh", "acbfgdhe", "acbfgedh", "acbfgehd", "acbfghde", "acbfghed", "acbfhdeg", "acbfhdge", "acbfhedg", "acbfhegd", "acbfhgde", "acbfhged", "acbgdefh", "acbgdehf", "acbgdfeh", "acbgdfhe", "acbgdhef", "acbgdhfe", "acbgedfh", "acbgedhf", "acbgefdh", "acbgefhd", "acbgehdf", "acbgehfd", "acbgfdeh", "acbgfdhe", "acbgfedh", "acbgfehd", "acbgfhde", "acbgfhed", "acbghdef", "acbghdfe", "acbghedf", "acbghefd", "acbghfde", "acbghfed", "acbhdefg", "acbhdegf", "acbhdfeg", "acbhdfge", "acbhdgef", "acbhdgfe", "acbhedfg", "acbhedgf", "acbhefdg", "acbhefgd", "acbhegdf", "acbhegfd", "acbhfdeg", "acbhfdge", "acbhfedg", "acbhfegd", "acbhfgde", "acbhfged", "acbhgdef", "acbhgdfe", "acbhgedf", "acbhgefd", "acbhgfde", "acbhgfed", "acdbefgh", "acdbefhg", "acdbegfh", "acdbeghf", "acdbehfg", "acdbehgf", "acdbfegh", "acdbfehg", "acdbfgeh", "acdbfghe", "acdbfheg", "acdbfhge", "acdbgefh", "acdbgehf", "acdbgfeh", "acdbgfhe", "acdbghef", "acdbghfe", "acdbhefg", "acdbhegf", "acdbhfeg", "acdbhfge", "acdbhgef", "acdbhgfe", "acdebfgh", "acdebfhg", "acdebgfh", "acdebghf", "acdebhfg", "acdebhgf", "acdefbgh", "acdefbhg", "acdefgbh", "acdefghb", "acdefhbg", "acdefhgb", "acdegbfh", "acdegbhf", "acdegfbh", "acdegfhb", "acdeghbf", "acdeghfb", "acdehbfg", "acdehbgf", "acdehfbg", "acdehfgb", "acdehgbf", "acdehgfb", "acdfbegh", "acdfbehg", "acdfbgeh", "acdfbghe", "acdfbheg", "acdfbhge", "acdfebgh", "acdfebhg", "acdfegbh", "acdfeghb", "acdfehbg", "acdfehgb", "acdfgbeh", "acdfgbhe", "acdfgebh", "acdfgehb", "acdfghbe", "acdfgheb", "acdfhbeg", "acdfhbge", "acdfhebg", "acdfhegb", "acdfhgbe", "acdfhgeb", "acdgbefh", "acdgbehf", "acdgbfeh", "acdgbfhe", "acdgbhef", "acdgbhfe", "acdgebfh", "acdgebhf", "acdgefbh", "acdgefhb", "acdgehbf", "acdgehfb", "acdgfbeh", "acdgfbhe", "acdgfebh", "acdgfehb", "acdgfhbe", "acdgfheb", "acdghbef", "acdghbfe", "acdghebf", "acdghefb", "acdghfbe", "acdghfeb", "acdhbefg", "acdhbegf", "acdhbfeg", "acdhbfge", "acdhbgef", "acdhbgfe", "acdhebfg", "acdhebgf", "acdhefbg", "acdhefgb", "acdhegbf", "acdhegfb", "acdhfbeg", "acdhfbge", "acdhfebg", "acdhfegb", "acdhfgbe", "acdhfgeb", "acdhgbef", "acdhgbfe", "acdhgebf", "acdhgefb", "acdhgfbe", "acdhgfeb", "acebdfgh", "acebdfhg", "acebdgfh", "acebdghf", "acebdhfg", "acebdhgf", "acebfdgh", "acebfdhg", "acebfgdh", "acebfghd", "acebfhdg", "acebfhgd", "acebgdfh", "acebgdhf", "acebgfdh", "acebgfhd", "acebghdf", "acebghfd", "acebhdfg", "acebhdgf", "acebhfdg", "acebhfgd", "acebhgdf", "acebhgfd", "acedbfgh", "acedbfhg", "acedbgfh", "acedbghf", "acedbhfg", "acedbhgf", "acedfbgh", "acedfbhg", "acedfgbh", "acedfghb", "acedfhbg", "acedfhgb", "acedgbfh", "acedgbhf", "acedgfbh", "acedgfhb", "acedghbf", "acedghfb", "acedhbfg", "acedhbgf", "acedhfbg", "acedhfgb", "acedhgbf", "acedhgfb", "acefbdgh", "acefbdhg", "acefbgdh", "acefbghd", "acefbhdg", "acefbhgd", "acefdbgh", "acefdbhg", "acefdgbh", "acefdghb", "acefdhbg", "acefdhgb", "acefgbdh", "acefgbhd", "acefgdbh", "acefgdhb", "acefghbd", "acefghdb", "acefhbdg", "acefhbgd", "acefhdbg", "acefhdgb", "acefhgbd", "acefhgdb", "acegbdfh", "acegbdhf", "acegbfdh", "acegbfhd", "acegbhdf", "acegbhfd", "acegdbfh", "acegdbhf", "acegdfbh", "acegdfhb", "acegdhbf", "acegdhfb", "acegfbdh", "acegfbhd", "acegfdbh", "acegfdhb", "acegfhbd", "acegfhdb", "aceghbdf", "aceghbfd", "aceghdbf", "aceghdfb", "aceghfbd", "aceghfdb", "acehbdfg", "acehbdgf", "acehbfdg", "acehbfgd", "acehbgdf", "acehbgfd", "acehdbfg", "acehdbgf", "acehdfbg", "acehdfgb", "acehdgbf", "acehdgfb", "acehfbdg", "acehfbgd", "acehfdbg", "acehfdgb", "acehfgbd", "acehfgdb", "acehgbdf", "acehgbfd", "acehgdbf", "acehgdfb", "acehgfbd", "acehgfdb", "acfbdegh", "acfbdehg", "acfbdgeh", "acfbdghe", "acfbdheg", "acfbdhge", "acfbedgh", "acfbedhg", "acfbegdh", "acfbeghd", "acfbehdg", "acfbehgd", "acfbgdeh", "acfbgdhe", "acfbgedh", "acfbgehd", "acfbghde", "acfbghed", "acfbhdeg", "acfbhdge", "acfbhedg", "acfbhegd", "acfbhgde", "acfbhged", "acfdbegh", "acfdbehg", "acfdbgeh", "acfdbghe", "acfdbheg", "acfdbhge", "acfdebgh", "acfdebhg", "acfdegbh", "acfdeghb", "acfdehbg", "acfdehgb", "acfdgbeh", "acfdgbhe", "acfdgebh", "acfdgehb", "acfdghbe", "acfdgheb", "acfdhbeg", "acfdhbge", "acfdhebg", "acfdhegb", "acfdhgbe", "acfdhgeb", "acfebdgh", "acfebdhg", "acfebgdh", "acfebghd", "acfebhdg", "acfebhgd", "acfedbgh", "acfedbhg", "acfedgbh", "acfedghb", "acfedhbg", "acfedhgb", "acfegbdh", "acfegbhd", "acfegdbh", "acfegdhb", "acfeghbd", "acfeghdb", "acfehbdg", "acfehbgd", "acfehdbg", "acfehdgb", "acfehgbd", "acfehgdb", "acfgbdeh", "acfgbdhe", "acfgbedh", "acfgbehd", "acfgbhde", "acfgbhed", "acfgdbeh", "acfgdbhe", "acfgdebh", "acfgdehb", "acfgdhbe", "acfgdheb", "acfgebdh", "acfgebhd", "acfgedbh", "acfgedhb", "acfgehbd", "acfgehdb", "acfghbde", "acfghbed", "acfghdbe", "acfghdeb", "acfghebd", "acfghedb", "acfhbdeg", "acfhbdge", "acfhbedg", "acfhbegd", "acfhbgde", "acfhbged", "acfhdbeg", "acfhdbge", "acfhdebg", "acfhdegb", "acfhdgbe", "acfhdgeb", "acfhebdg", "acfhebgd", "acfhedbg", "acfhedgb", "acfhegbd", "acfhegdb", "acfhgbde", "acfhgbed", "acfhgdbe", "acfhgdeb", "acfhgebd", "acfhgedb", "acgbdefh", "acgbdehf", "acgbdfeh", "acgbdfhe", "acgbdhef", "acgbdhfe", "acgbedfh", "acgbedhf", "acgbefdh", "acgbefhd", "acgbehdf", "acgbehfd", "acgbfdeh", "acgbfdhe", "acgbfedh", "acgbfehd", "acgbfhde", "acgbfhed", "acgbhdef", "acgbhdfe", "acgbhedf", "acgbhefd", "acgbhfde", "acgbhfed", "acgdbefh", "acgdbehf", "acgdbfeh", "acgdbfhe", "acgdbhef", "acgdbhfe", "acgdebfh", "acgdebhf", "acgdefbh", "acgdefhb", "acgdehbf", "acgdehfb", "acgdfbeh", "acgdfbhe", "acgdfebh", "acgdfehb", "acgdfhbe", "acgdfheb", "acgdhbef", "acgdhbfe", "acgdhebf", "acgdhefb", "acgdhfbe", "acgdhfeb", "acgebdfh", "acgebdhf", "acgebfdh", "acgebfhd", "acgebhdf", "acgebhfd", "acgedbfh", "acgedbhf", "acgedfbh", "acgedfhb", "acgedhbf", "acgedhfb", "acgefbdh", "acgefbhd", "acgefdbh", "acgefdhb", "acgefhbd", "acgefhdb", "acgehbdf", "acgehbfd", "acgehdbf", "acgehdfb", "acgehfbd", "acgehfdb", "acgfbdeh", "acgfbdhe", "acgfbedh", "acgfbehd", "acgfbhde", "acgfbhed", "acgfdbeh", "acgfdbhe", "acgfdebh", "acgfdehb", "acgfdhbe", "acgfdheb", "acgfebdh", "acgfebhd", "acgfedbh", "acgfedhb", "acgfehbd", "acgfehdb", "acgfhbde", "acgfhbed", "acgfhdbe", "acgfhdeb", "acgfhebd", "acgfhedb", "acghbdef", "acghbdfe", "acghbedf", "acghbefd", "acghbfde", "acghbfed", "acghdbef", "acghdbfe", "acghdebf", "acghdefb", "acghdfbe", "acghdfeb", "acghebdf", "acghebfd", "acghedbf", "acghedfb", "acghefbd", "acghefdb", "acghfbde", "acghfbed", "acghfdbe", "acghfdeb", "acghfebd", "acghfedb", "achbdefg", "achbdegf", "achbdfeg", "achbdfge", "achbdgef", "achbdgfe", "achbedfg", "achbedgf", "achbefdg", "achbefgd", "achbegdf", "achbegfd", "achbfdeg", "achbfdge", "achbfedg", "achbfegd", "achbfgde", "achbfged", "achbgdef", "achbgdfe", "achbgedf", "achbgefd", "achbgfde", "achbgfed", "achdbefg", "achdbegf", "achdbfeg", "achdbfge", "achdbgef", "achdbgfe", "achdebfg", "achdebgf", "achdefbg", "achdefgb", "achdegbf", "achdegfb", "achdfbeg", "achdfbge", "achdfebg", "achdfegb", "achdfgbe", "achdfgeb", "achdgbef", "achdgbfe", "achdgebf", "achdgefb", "achdgfbe", "achdgfeb", "achebdfg", "achebdgf", "achebfdg", "achebfgd", "achebgdf", "achebgfd", "achedbfg", "achedbgf", "achedfbg", "achedfgb", "achedgbf", "achedgfb", "achefbdg", "achefbgd", "achefdbg", "achefdgb", "achefgbd", "achefgdb", "achegbdf", "achegbfd", "achegdbf", "achegdfb", "achegfbd", "achegfdb", "achfbdeg", "achfbdge", "achfbedg", "achfbegd", "achfbgde", "achfbged", "achfdbeg", "achfdbge", "achfdebg", "achfdegb", "achfdgbe", "achfdgeb", "achfebdg", "achfebgd", "achfedbg", "achfedgb", "achfegbd", "achfegdb", "achfgbde", "achfgbed", "achfgdbe", "achfgdeb", "achfgebd", "achfgedb", "achgbdef", "achgbdfe", "achgbedf", "achgbefd", "achgbfde", "achgbfed", "achgdbef", "achgdbfe", "achgdebf", "achgdefb", "achgdfbe", "achgdfeb", "achgebdf", "achgebfd", "achgedbf", "achgedfb", "achgefbd", "achgefdb", "achgfbde", "achgfbed", "achgfdbe", "achgfdeb", "achgfebd", "achgfedb", "adbcefgh", "adbcefhg", "adbcegfh", "adbceghf", "adbcehfg", "adbcehgf", "adbcfegh", "adbcfehg", "adbcfgeh", "adbcfghe", "adbcfheg", "adbcfhge", "adbcgefh", "adbcgehf", "adbcgfeh", "adbcgfhe", "adbcghef", "adbcghfe", "adbchefg", "adbchegf", "adbchfeg", "adbchfge", "adbchgef", "adbchgfe", "adbecfgh", "adbecfhg", "adbecgfh", "adbecghf", "adbechfg", "adbechgf", "adbefcgh", "adbefchg", "adbefgch", "adbefghc", "adbefhcg", "adbefhgc", "adbegcfh", "adbegchf", "adbegfch", "adbegfhc", "adbeghcf", "adbeghfc", "adbehcfg", "adbehcgf", "adbehfcg", "adbehfgc", "adbehgcf", "adbehgfc", "adbfcegh", "adbfcehg", "adbfcgeh", "adbfcghe", "adbfcheg", "adbfchge", "adbfecgh", "adbfechg", "adbfegch", "adbfeghc", "adbfehcg", "adbfehgc", "adbfgceh", "adbfgche", "adbfgech", "adbfgehc", "adbfghce", "adbfghec", "adbfhceg", "adbfhcge", "adbfhecg", "adbfhegc", "adbfhgce", "adbfhgec", "adbgcefh", "adbgcehf", "adbgcfeh", "adbgcfhe", "adbgchef", "adbgchfe", "adbgecfh", "adbgechf", "adbgefch", "adbgefhc", "adbgehcf", "adbgehfc", "adbgfceh", "adbgfche", "adbgfech", "adbgfehc", "adbgfhce", "adbgfhec", "adbghcef", "adbghcfe", "adbghecf", "adbghefc", "adbghfce", "adbghfec", "adbhcefg", "adbhcegf", "adbhcfeg", "adbhcfge", "adbhcgef", "adbhcgfe", "adbhecfg", "adbhecgf", "adbhefcg", "adbhefgc", "adbhegcf", "adbhegfc", "adbhfceg", "adbhfcge", "adbhfecg", "adbhfegc", "adbhfgce", "adbhfgec", "adbhgcef", "adbhgcfe", "adbhgecf", "adbhgefc", "adbhgfce", "adbhgfec", "adcbefgh", "adcbefhg", "adcbegfh", "adcbeghf", "adcbehfg", "adcbehgf", "adcbfegh", "adcbfehg", "adcbfgeh", "adcbfghe", "adcbfheg", "adcbfhge", "adcbgefh", "adcbgehf", "adcbgfeh", "adcbgfhe", "adcbghef", "adcbghfe", "adcbhefg", "adcbhegf", "adcbhfeg", "adcbhfge", "adcbhgef", "adcbhgfe", "adcebfgh", "adcebfhg", "adcebgfh", "adcebghf", "adcebhfg", "adcebhgf", "adcefbgh", "adcefbhg", "adcefgbh", "adcefghb", "adcefhbg", "adcefhgb", "adcegbfh", "adcegbhf", "adcegfbh", "adcegfhb", "adceghbf", "adceghfb", "adcehbfg", "adcehbgf", "adcehfbg", "adcehfgb", "adcehgbf", "adcehgfb", "adcfbegh", "adcfbehg", "adcfbgeh", "adcfbghe", "adcfbheg", "adcfbhge", "adcfebgh", "adcfebhg", "adcfegbh", "adcfeghb", "adcfehbg", "adcfehgb", "adcfgbeh", "adcfgbhe", "adcfgebh", "adcfgehb", "adcfghbe", "adcfgheb", "adcfhbeg", "adcfhbge", "adcfhebg", "adcfhegb", "adcfhgbe", "adcfhgeb", "adcgbefh", "adcgbehf", "adcgbfeh", "adcgbfhe", "adcgbhef", "adcgbhfe", "adcgebfh", "adcgebhf", "adcgefbh", "adcgefhb", "adcgehbf", "adcgehfb", "adcgfbeh", "adcgfbhe", "adcgfebh", "adcgfehb", "adcgfhbe", "adcgfheb", "adcghbef", "adcghbfe", "adcghebf", "adcghefb", "adcghfbe", "adcghfeb", "adchbefg", "adchbegf", "adchbfeg", "adchbfge", "adchbgef", "adchbgfe", "adchebfg", "adchebgf", "adchefbg", "adchefgb", "adchegbf", "adchegfb", "adchfbeg", "adchfbge", "adchfebg", "adchfegb", "adchfgbe", "adchfgeb", "adchgbef", "adchgbfe", "adchgebf", "adchgefb", "adchgfbe", "adchgfeb", "adebcfgh", "adebcfhg", "adebcgfh", "adebcghf", "adebchfg", "adebchgf", "adebfcgh", "adebfchg", "adebfgch", "adebfghc", "adebfhcg", "adebfhgc", "adebgcfh", "adebgchf", "adebgfch", "adebgfhc", "adebghcf", "adebghfc", "adebhcfg", "adebhcgf", "adebhfcg", "adebhfgc", "adebhgcf", "adebhgfc", "adecbfgh", "adecbfhg", "adecbgfh", "adecbghf", "adecbhfg", "adecbhgf", "adecfbgh", "adecfbhg", "adecfgbh", "adecfghb", "adecfhbg", "adecfhgb", "adecgbfh", "adecgbhf", "adecgfbh", "adecgfhb", "adecghbf", "adecghfb", "adechbfg", "adechbgf", "adechfbg", "adechfgb", "adechgbf", "adechgfb", "adefbcgh", "adefbchg", "adefbgch", "adefbghc", "adefbhcg", "adefbhgc", "adefcbgh", "adefcbhg", "adefcgbh", "adefcghb", "adefchbg", "adefchgb", "adefgbch", "adefgbhc", "adefgcbh", "adefgchb", "adefghbc", "adefghcb", "adefhbcg", "adefhbgc", "adefhcbg", "adefhcgb", "adefhgbc", "adefhgcb", "adegbcfh", "adegbchf", "adegbfch", "adegbfhc", "adegbhcf", "adegbhfc", "adegcbfh", "adegcbhf", "adegcfbh", "adegcfhb", "adegchbf", "adegchfb", "adegfbch", "adegfbhc", "adegfcbh", "adegfchb", "adegfhbc", "adegfhcb", "adeghbcf", "adeghbfc", "adeghcbf", "adeghcfb", "adeghfbc", "adeghfcb", "adehbcfg", "adehbcgf", "adehbfcg", "adehbfgc", "adehbgcf", "adehbgfc", "adehcbfg", "adehcbgf", "adehcfbg", "adehcfgb", "adehcgbf", "adehcgfb", "adehfbcg", "adehfbgc", "adehfcbg", "adehfcgb", "adehfgbc", "adehfgcb", "adehgbcf", "adehgbfc", "adehgcbf", "adehgcfb", "adehgfbc", "adehgfcb", "adfbcegh", "adfbcehg", "adfbcgeh", "adfbcghe", "adfbcheg", "adfbchge", "adfbecgh", "adfbechg", "adfbegch", "adfbeghc", "adfbehcg", "adfbehgc", "adfbgceh", "adfbgche", "adfbgech", "adfbgehc", "adfbghce", "adfbghec", "adfbhceg", "adfbhcge", "adfbhecg", "adfbhegc", "adfbhgce", "adfbhgec", "adfcbegh", "adfcbehg", "adfcbgeh", "adfcbghe", "adfcbheg", "adfcbhge", "adfcebgh", "adfcebhg", "adfcegbh", "adfceghb", "adfcehbg", "adfcehgb", "adfcgbeh", "adfcgbhe", "adfcgebh", "adfcgehb", "adfcghbe", "adfcgheb", "adfchbeg", "adfchbge", "adfchebg", "adfchegb", "adfchgbe", "adfchgeb", "adfebcgh", "adfebchg", "adfebgch", "adfebghc", "adfebhcg", "adfebhgc", "adfecbgh", "adfecbhg", "adfecgbh", "adfecghb", "adfechbg", "adfechgb", "adfegbch", "adfegbhc", "adfegcbh", "adfegchb", "adfeghbc", "adfeghcb", "adfehbcg", "adfehbgc", "adfehcbg", "adfehcgb", "adfehgbc", "adfehgcb", "adfgbceh", "adfgbche", "adfgbech", "adfgbehc", "adfgbhce", "adfgbhec", "adfgcbeh", "adfgcbhe", "adfgcebh", "adfgcehb", "adfgchbe", "adfgcheb", "adfgebch", "adfgebhc", "adfgecbh", "adfgechb", "adfgehbc", "adfgehcb", "adfghbce", "adfghbec", "adfghcbe", "adfghceb", "adfghebc", "adfghecb", "adfhbceg", "adfhbcge", "adfhbecg", "adfhbegc", "adfhbgce", "adfhbgec", "adfhcbeg", "adfhcbge", "adfhcebg", "adfhcegb", "adfhcgbe", "adfhcgeb", "adfhebcg", "adfhebgc", "adfhecbg", "adfhecgb", "adfhegbc", "adfhegcb", "adfhgbce", "adfhgbec", "adfhgcbe", "adfhgceb", "adfhgebc", "adfhgecb", "adgbcefh", "adgbcehf", "adgbcfeh", "adgbcfhe", "adgbchef", "adgbchfe", "adgbecfh", "adgbechf", "adgbefch", "adgbefhc", "adgbehcf", "adgbehfc", "adgbfceh", "adgbfche", "adgbfech", "adgbfehc", "adgbfhce", "adgbfhec", "adgbhcef", "adgbhcfe", "adgbhecf", "adgbhefc", "adgbhfce", "adgbhfec", "adgcbefh", "adgcbehf", "adgcbfeh", "adgcbfhe", "adgcbhef", "adgcbhfe", "adgcebfh", "adgcebhf", "adgcefbh", "adgcefhb", "adgcehbf", "adgcehfb", "adgcfbeh", "adgcfbhe", "adgcfebh", "adgcfehb", "adgcfhbe", "adgcfheb", "adgchbef", "adgchbfe", "adgchebf", "adgchefb", "adgchfbe", "adgchfeb", "adgebcfh", "adgebchf", "adgebfch", "adgebfhc", "adgebhcf", "adgebhfc", "adgecbfh", "adgecbhf", "adgecfbh", "adgecfhb", "adgechbf", "adgechfb", "adgefbch", "adgefbhc", "adgefcbh", "adgefchb", "adgefhbc", "adgefhcb", "adgehbcf", "adgehbfc", "adgehcbf", "adgehcfb", "adgehfbc", "adgehfcb", "adgfbceh", "adgfbche", "adgfbech", "adgfbehc", "adgfbhce", "adgfbhec", "adgfcbeh", "adgfcbhe", "adgfcebh", "adgfcehb", "adgfchbe", "adgfcheb", "adgfebch", "adgfebhc", "adgfecbh", "adgfechb", "adgfehbc", "adgfehcb", "adgfhbce", "adgfhbec", "adgfhcbe", "adgfhceb", "adgfhebc", "adgfhecb", "adghbcef", "adghbcfe", "adghbecf", "adghbefc", "adghbfce", "adghbfec", "adghcbef", "adghcbfe", "adghcebf", "adghcefb", "adghcfbe", "adghcfeb", "adghebcf", "adghebfc", "adghecbf", "adghecfb", "adghefbc", "adghefcb", "adghfbce", "adghfbec", "adghfcbe", "adghfceb", "adghfebc", "adghfecb", "adhbcefg", "adhbcegf", "adhbcfeg", "adhbcfge", "adhbcgef", "adhbcgfe", "adhbecfg", "adhbecgf", "adhbefcg", "adhbefgc", "adhbegcf", "adhbegfc", "adhbfceg", "adhbfcge", "adhbfecg", "adhbfegc", "adhbfgce", "adhbfgec", "adhbgcef", "adhbgcfe", "adhbgecf", "adhbgefc", "adhbgfce", "adhbgfec", "adhcbefg", "adhcbegf", "adhcbfeg", "adhcbfge", "adhcbgef", "adhcbgfe", "adhcebfg", "adhcebgf", "adhcefbg", "adhcefgb", "adhcegbf", "adhcegfb", "adhcfbeg", "adhcfbge", "adhcfebg", "adhcfegb", "adhcfgbe", "adhcfgeb", "adhcgbef", "adhcgbfe", "adhcgebf", "adhcgefb", "adhcgfbe", "adhcgfeb", "adhebcfg", "adhebcgf", "adhebfcg", "adhebfgc", "adhebgcf", "adhebgfc", "adhecbfg", "adhecbgf", "adhecfbg", "adhecfgb", "adhecgbf", "adhecgfb", "adhefbcg", "adhefbgc", "adhefcbg", "adhefcgb", "adhefgbc", "adhefgcb", "adhegbcf", "adhegbfc", "adhegcbf", "adhegcfb", "adhegfbc", "adhegfcb", "adhfbceg", "adhfbcge", "adhfbecg", "adhfbegc", "adhfbgce", "adhfbgec", "adhfcbeg", "adhfcbge", "adhfcebg", "adhfcegb", "adhfcgbe", "adhfcgeb", "adhfebcg", "adhfebgc", "adhfecbg", "adhfecgb", "adhfegbc", "adhfegcb", "adhfgbce", "adhfgbec", "adhfgcbe", "adhfgceb", "adhfgebc", "adhfgecb", "adhgbcef", "adhgbcfe", "adhgbecf", "adhgbefc", "adhgbfce", "adhgbfec", "adhgcbef", "adhgcbfe", "adhgcebf", "adhgcefb", "adhgcfbe", "adhgcfeb", "adhgebcf", "adhgebfc", "adhgecbf", "adhgecfb", "adhgefbc", "adhgefcb", "adhgfbce", "adhgfbec", "adhgfcbe", "adhgfceb", "adhgfebc", "adhgfecb", "aebcdfgh", "aebcdfhg", "aebcdgfh", "aebcdghf", "aebcdhfg", "aebcdhgf", "aebcfdgh", "aebcfdhg", "aebcfgdh", "aebcfghd", "aebcfhdg", "aebcfhgd", "aebcgdfh", "aebcgdhf", "aebcgfdh", "aebcgfhd", "aebcghdf", "aebcghfd", "aebchdfg", "aebchdgf", "aebchfdg", "aebchfgd", "aebchgdf", "aebchgfd", "aebdcfgh", "aebdcfhg", "aebdcgfh", "aebdcghf", "aebdchfg", "aebdchgf", "aebdfcgh", "aebdfchg", "aebdfgch", "aebdfghc", "aebdfhcg", "aebdfhgc", "aebdgcfh", "aebdgchf", "aebdgfch", "aebdgfhc", "aebdghcf", "aebdghfc", "aebdhcfg", "aebdhcgf", "aebdhfcg", "aebdhfgc", "aebdhgcf", "aebdhgfc", "aebfcdgh", "aebfcdhg", "aebfcgdh", "aebfcghd", "aebfchdg", "aebfchgd", "aebfdcgh", "aebfdchg", "aebfdgch", "aebfdghc", "aebfdhcg", "aebfdhgc", "aebfgcdh", "aebfgchd", "aebfgdch", "aebfgdhc", "aebfghcd", "aebfghdc", "aebfhcdg", "aebfhcgd", "aebfhdcg", "aebfhdgc", "aebfhgcd", "aebfhgdc", "aebgcdfh", "aebgcdhf", "aebgcfdh", "aebgcfhd", "aebgchdf", "aebgchfd", "aebgdcfh", "aebgdchf", "aebgdfch", "aebgdfhc", "aebgdhcf", "aebgdhfc", "aebgfcdh", "aebgfchd", "aebgfdch", "aebgfdhc", "aebgfhcd", "aebgfhdc", "aebghcdf", "aebghcfd", "aebghdcf", "aebghdfc", "aebghfcd", "aebghfdc", "aebhcdfg", "aebhcdgf", "aebhcfdg", "aebhcfgd", "aebhcgdf", "aebhcgfd", "aebhdcfg", "aebhdcgf", "aebhdfcg", "aebhdfgc", "aebhdgcf", "aebhdgfc", "aebhfcdg", "aebhfcgd", "aebhfdcg", "aebhfdgc", "aebhfgcd", "aebhfgdc", "aebhgcdf", "aebhgcfd", "aebhgdcf", "aebhgdfc", "aebhgfcd", "aebhgfdc", "aecbdfgh", "aecbdfhg", "aecbdgfh", "aecbdghf", "aecbdhfg", "aecbdhgf", "aecbfdgh", "aecbfdhg", "aecbfgdh", "aecbfghd", "aecbfhdg", "aecbfhgd", "aecbgdfh", "aecbgdhf", "aecbgfdh", "aecbgfhd", "aecbghdf", "aecbghfd", "aecbhdfg", "aecbhdgf", "aecbhfdg", "aecbhfgd", "aecbhgdf", "aecbhgfd", "aecdbfgh", "aecdbfhg", "aecdbgfh", "aecdbghf", "aecdbhfg", "aecdbhgf", "aecdfbgh", "aecdfbhg", "aecdfgbh", "aecdfghb", "aecdfhbg", "aecdfhgb", "aecdgbfh", "aecdgbhf", "aecdgfbh", "aecdgfhb", "aecdghbf", "aecdghfb", "aecdhbfg", "aecdhbgf", "aecdhfbg", "aecdhfgb", "aecdhgbf", "aecdhgfb", "aecfbdgh", "aecfbdhg", "aecfbgdh", "aecfbghd", "aecfbhdg", "aecfbhgd", "aecfdbgh", "aecfdbhg", "aecfdgbh", "aecfdghb", "aecfdhbg", "aecfdhgb", "aecfgbdh", "aecfgbhd", "aecfgdbh", "aecfgdhb", "aecfghbd", "aecfghdb", "aecfhbdg", "aecfhbgd", "aecfhdbg", "aecfhdgb", "aecfhgbd", "aecfhgdb", "aecgbdfh", "aecgbdhf", "aecgbfdh", "aecgbfhd", "aecgbhdf", "aecgbhfd", "aecgdbfh", "aecgdbhf", "aecgdfbh", "aecgdfhb", "aecgdhbf", "aecgdhfb", "aecgfbdh", "aecgfbhd", "aecgfdbh", "aecgfdhb", "aecgfhbd", "aecgfhdb", "aecghbdf", "aecghbfd", "aecghdbf", "aecghdfb", "aecghfbd", "aecghfdb", "aechbdfg", "aechbdgf", "aechbfdg", "aechbfgd", "aechbgdf", "aechbgfd", "aechdbfg", "aechdbgf", "aechdfbg", "aechdfgb", "aechdgbf", "aechdgfb", "aechfbdg", "aechfbgd", "aechfdbg", "aechfdgb", "aechfgbd", "aechfgdb", "aechgbdf", "aechgbfd", "aechgdbf", "aechgdfb", "aechgfbd", "aechgfdb", "aedbcfgh", "aedbcfhg", "aedbcgfh", "aedbcghf", "aedbchfg", "aedbchgf", "aedbfcgh", "aedbfchg", "aedbfgch", "aedbfghc", "aedbfhcg", "aedbfhgc", "aedbgcfh", "aedbgchf", "aedbgfch", "aedbgfhc", "aedbghcf", "aedbghfc", "aedbhcfg", "aedbhcgf", "aedbhfcg", "aedbhfgc", "aedbhgcf", "aedbhgfc", "aedcbfgh", "aedcbfhg", "aedcbgfh", "aedcbghf", "aedcbhfg", "aedcbhgf", "aedcfbgh", "aedcfbhg", "aedcfgbh", "aedcfghb", "aedcfhbg", "aedcfhgb", "aedcgbfh", "aedcgbhf", "aedcgfbh", "aedcgfhb", "aedcghbf", "aedcghfb", "aedchbfg", "aedchbgf", "aedchfbg", "aedchfgb", "aedchgbf", "aedchgfb", "aedfbcgh", "aedfbchg", "aedfbgch", "aedfbghc", "aedfbhcg", "aedfbhgc", "aedfcbgh", "aedfcbhg", "aedfcgbh", "aedfcghb", "aedfchbg", "aedfchgb", "aedfgbch", "aedfgbhc", "aedfgcbh", "aedfgchb", "aedfghbc", "aedfghcb", "aedfhbcg", "aedfhbgc", "aedfhcbg", "aedfhcgb", "aedfhgbc", "aedfhgcb", "aedgbcfh", "aedgbchf", "aedgbfch", "aedgbfhc", "aedgbhcf", "aedgbhfc", "aedgcbfh", "aedgcbhf", "aedgcfbh", "aedgcfhb", "aedgchbf", "aedgchfb", "aedgfbch", "aedgfbhc", "aedgfcbh", "aedgfchb", "aedgfhbc", "aedgfhcb", "aedghbcf", "aedghbfc", "aedghcbf", "aedghcfb", "aedghfbc", "aedghfcb", "aedhbcfg", "aedhbcgf", "aedhbfcg", "aedhbfgc", "aedhbgcf", "aedhbgfc", "aedhcbfg", "aedhcbgf", "aedhcfbg", "aedhcfgb", "aedhcgbf", "aedhcgfb", "aedhfbcg", "aedhfbgc", "aedhfcbg", "aedhfcgb", "aedhfgbc", "aedhfgcb", "aedhgbcf", "aedhgbfc", "aedhgcbf", "aedhgcfb", "aedhgfbc", "aedhgfcb", "aefbcdgh", "aefbcdhg", "aefbcgdh", "aefbcghd", "aefbchdg", "aefbchgd", "aefbdcgh", "aefbdchg", "aefbdgch", "aefbdghc", "aefbdhcg", "aefbdhgc", "aefbgcdh", "aefbgchd", "aefbgdch", "aefbgdhc", "aefbghcd", "aefbghdc", "aefbhcdg", "aefbhcgd", "aefbhdcg", "aefbhdgc", "aefbhgcd", "aefbhgdc", "aefcbdgh", "aefcbdhg", "aefcbgdh", "aefcbghd", "aefcbhdg", "aefcbhgd", "aefcdbgh", "aefcdbhg", "aefcdgbh", "aefcdghb", "aefcdhbg", "aefcdhgb", "aefcgbdh", "aefcgbhd", "aefcgdbh", "aefcgdhb", "aefcghbd", "aefcghdb", "aefchbdg", "aefchbgd", "aefchdbg", "aefchdgb", "aefchgbd", "aefchgdb", "aefdbcgh", "aefdbchg", "aefdbgch", "aefdbghc", "aefdbhcg", "aefdbhgc", "aefdcbgh", "aefdcbhg", "aefdcgbh", "aefdcghb", "aefdchbg", "aefdchgb", "aefdgbch", "aefdgbhc", "aefdgcbh", "aefdgchb", "aefdghbc", "aefdghcb", "aefdhbcg", "aefdhbgc", "aefdhcbg", "aefdhcgb", "aefdhgbc", "aefdhgcb", "aefgbcdh", "aefgbchd", "aefgbdch", "aefgbdhc", "aefgbhcd", "aefgbhdc", "aefgcbdh", "aefgcbhd", "aefgcdbh", "aefgcdhb", "aefgchbd", "aefgchdb", "aefgdbch", "aefgdbhc", "aefgdcbh", "aefgdchb", "aefgdhbc", "aefgdhcb", "aefghbcd", "aefghbdc", "aefghcbd", "aefghcdb", "aefghdbc", "aefghdcb", "aefhbcdg", "aefhbcgd", "aefhbdcg", "aefhbdgc", "aefhbgcd", "aefhbgdc", "aefhcbdg", "aefhcbgd", "aefhcdbg", "aefhcdgb", "aefhcgbd", "aefhcgdb", "aefhdbcg", "aefhdbgc", "aefhdcbg", "aefhdcgb", "aefhdgbc", "aefhdgcb", "aefhgbcd", "aefhgbdc", "aefhgcbd", "aefhgcdb", "aefhgdbc", "aefhgdcb", "aegbcdfh", "aegbcdhf", "aegbcfdh", "aegbcfhd", "aegbchdf", "aegbchfd", "aegbdcfh", "aegbdchf", "aegbdfch", "aegbdfhc", "aegbdhcf", "aegbdhfc", "aegbfcdh", "aegbfchd", "aegbfdch", "aegbfdhc", "aegbfhcd", "aegbfhdc", "aegbhcdf", "aegbhcfd", "aegbhdcf", "aegbhdfc", "aegbhfcd", "aegbhfdc", "aegcbdfh", "aegcbdhf", "aegcbfdh", "aegcbfhd", "aegcbhdf", "aegcbhfd", "aegcdbfh", "aegcdbhf", "aegcdfbh", "aegcdfhb", "aegcdhbf", "aegcdhfb", "aegcfbdh", "aegcfbhd", "aegcfdbh", "aegcfdhb", "aegcfhbd", "aegcfhdb", "aegchbdf", "aegchbfd", "aegchdbf", "aegchdfb", "aegchfbd", "aegchfdb", "aegdbcfh", "aegdbchf", "aegdbfch", "aegdbfhc", "aegdbhcf", "aegdbhfc", "aegdcbfh", "aegdcbhf", "aegdcfbh", "aegdcfhb", "aegdchbf", "aegdchfb", "aegdfbch", "aegdfbhc", "aegdfcbh", "aegdfchb", "aegdfhbc", "aegdfhcb", "aegdhbcf", "aegdhbfc", "aegdhcbf", "aegdhcfb", "aegdhfbc", "aegdhfcb", "aegfbcdh", "aegfbchd", "aegfbdch", "aegfbdhc", "aegfbhcd", "aegfbhdc", "aegfcbdh", "aegfcbhd", "aegfcdbh", "aegfcdhb", "aegfchbd", "aegfchdb", "aegfdbch", "aegfdbhc", "aegfdcbh", "aegfdchb", "aegfdhbc", "aegfdhcb", "aegfhbcd", "aegfhbdc", "aegfhcbd", "aegfhcdb", "aegfhdbc", "aegfhdcb", "aeghbcdf", "aeghbcfd", "aeghbdcf", "aeghbdfc", "aeghbfcd", "aeghbfdc", "aeghcbdf", "aeghcbfd", "aeghcdbf", "aeghcdfb", "aeghcfbd", "aeghcfdb", "aeghdbcf", "aeghdbfc", "aeghdcbf", "aeghdcfb", "aeghdfbc", "aeghdfcb", "aeghfbcd", "aeghfbdc", "aeghfcbd", "aeghfcdb", "aeghfdbc", "aeghfdcb", "aehbcdfg", "aehbcdgf", "aehbcfdg", "aehbcfgd", "aehbcgdf", "aehbcgfd", "aehbdcfg", "aehbdcgf", "aehbdfcg", "aehbdfgc", "aehbdgcf", "aehbdgfc", "aehbfcdg", "aehbfcgd", "aehbfdcg", "aehbfdgc", "aehbfgcd", "aehbfgdc", "aehbgcdf", "aehbgcfd", "aehbgdcf", "aehbgdfc", "aehbgfcd", "aehbgfdc", "aehcbdfg", "aehcbdgf", "aehcbfdg", "aehcbfgd", "aehcbgdf", "aehcbgfd", "aehcdbfg", "aehcdbgf", "aehcdfbg", "aehcdfgb", "aehcdgbf", "aehcdgfb", "aehcfbdg", "aehcfbgd", "aehcfdbg", "aehcfdgb", "aehcfgbd", "aehcfgdb", "aehcgbdf", "aehcgbfd", "aehcgdbf", "aehcgdfb", "aehcgfbd", "aehcgfdb", "aehdbcfg", "aehdbcgf", "aehdbfcg", "aehdbfgc", "aehdbgcf", "aehdbgfc", "aehdcbfg", "aehdcbgf", "aehdcfbg", "aehdcfgb", "aehdcgbf", "aehdcgfb", "aehdfbcg", "aehdfbgc", "aehdfcbg", "aehdfcgb", "aehdfgbc", "aehdfgcb", "aehdgbcf", "aehdgbfc", "aehdgcbf", "aehdgcfb", "aehdgfbc", "aehdgfcb", "aehfbcdg", "aehfbcgd", "aehfbdcg", "aehfbdgc", "aehfbgcd", "aehfbgdc", "aehfcbdg", "aehfcbgd", "aehfcdbg", "aehfcdgb", "aehfcgbd", "aehfcgdb", "aehfdbcg", "aehfdbgc", "aehfdcbg", "aehfdcgb", "aehfdgbc", "aehfdgcb", "aehfgbcd", "aehfgbdc", "aehfgcbd", "aehfgcdb", "aehfgdbc", "aehfgdcb", "aehgbcdf", "aehgbcfd", "aehgbdcf", "aehgbdfc", "aehgbfcd", "aehgbfdc", "aehgcbdf", "aehgcbfd", "aehgcdbf", "aehgcdfb", "aehgcfbd", "aehgcfdb", "aehgdbcf", "aehgdbfc", "aehgdcbf", "aehgdcfb", "aehgdfbc", "aehgdfcb", "aehgfbcd", "aehgfbdc", "aehgfcbd", "aehgfcdb", "aehgfdbc", "aehgfdcb", "afbcdegh", "afbcdehg", "afbcdgeh", "afbcdghe", "afbcdheg", "afbcdhge", "afbcedgh", "afbcedhg", "afbcegdh", "afbceghd", "afbcehdg", "afbcehgd", "afbcgdeh", "afbcgdhe", "afbcgedh", "afbcgehd", "afbcghde", "afbcghed", "afbchdeg", "afbchdge", "afbchedg", "afbchegd", "afbchgde", "afbchged", "afbdcegh", "afbdcehg", "afbdcgeh", "afbdcghe", "afbdcheg", "afbdchge", "afbdecgh", "afbdechg", "afbdegch", "afbdeghc", "afbdehcg", "afbdehgc", "afbdgceh", "afbdgche", "afbdgech", "afbdgehc", "afbdghce", "afbdghec", "afbdhceg", "afbdhcge", "afbdhecg", "afbdhegc", "afbdhgce", "afbdhgec", "afbecdgh", "afbecdhg", "afbecgdh", "afbecghd", "afbechdg", "afbechgd", "afbedcgh", "afbedchg", "afbedgch", "afbedghc", "afbedhcg", "afbedhgc", "afbegcdh", "afbegchd", "afbegdch", "afbegdhc", "afbeghcd", "afbeghdc", "afbehcdg", "afbehcgd", "afbehdcg", "afbehdgc", "afbehgcd", "afbehgdc", "afbgcdeh", "afbgcdhe", "afbgcedh", "afbgcehd", "afbgchde", "afbgched", "afbgdceh", "afbgdche", "afbgdech", "afbgdehc", "afbgdhce", "afbgdhec", "afbgecdh", "afbgechd", "afbgedch", "afbgedhc", "afbgehcd", "afbgehdc", "afbghcde", "afbghced", "afbghdce", "afbghdec", "afbghecd", "afbghedc", "afbhcdeg", "afbhcdge", "afbhcedg", "afbhcegd", "afbhcgde", "afbhcged", "afbhdceg", "afbhdcge", "afbhdecg", "afbhdegc", "afbhdgce", "afbhdgec", "afbhecdg", "afbhecgd", "afbhedcg", "afbhedgc", "afbhegcd", "afbhegdc", "afbhgcde", "afbhgced", "afbhgdce", "afbhgdec", "afbhgecd", "afbhgedc", "afcbdegh", "afcbdehg", "afcbdgeh", "afcbdghe", "afcbdheg", "afcbdhge", "afcbedgh", "afcbedhg", "afcbegdh", "afcbeghd", "afcbehdg", "afcbehgd", "afcbgdeh", "afcbgdhe", "afcbgedh", "afcbgehd", "afcbghde", "afcbghed", "afcbhdeg", "afcbhdge", "afcbhedg", "afcbhegd", "afcbhgde", "afcbhged", "afcdbegh", "afcdbehg", "afcdbgeh", "afcdbghe", "afcdbheg", "afcdbhge", "afcdebgh", "afcdebhg", "afcdegbh", "afcdeghb", "afcdehbg", "afcdehgb", "afcdgbeh", "afcdgbhe", "afcdgebh", "afcdgehb", "afcdghbe", "afcdgheb", "afcdhbeg", "afcdhbge", "afcdhebg", "afcdhegb", "afcdhgbe", "afcdhgeb", "afcebdgh", "afcebdhg", "afcebgdh", "afcebghd", "afcebhdg", "afcebhgd", "afcedbgh", "afcedbhg", "afcedgbh", "afcedghb", "afcedhbg", "afcedhgb", "afcegbdh", "afcegbhd", "afcegdbh", "afcegdhb", "afceghbd", "afceghdb", "afcehbdg", "afcehbgd", "afcehdbg", "afcehdgb", "afcehgbd", "afcehgdb", "afcgbdeh", "afcgbdhe", "afcgbedh", "afcgbehd", "afcgbhde", "afcgbhed", "afcgdbeh", "afcgdbhe", "afcgdebh", "afcgdehb", "afcgdhbe", "afcgdheb", "afcgebdh", "afcgebhd", "afcgedbh", "afcgedhb", "afcgehbd", "afcgehdb", "afcghbde", "afcghbed", "afcghdbe", "afcghdeb", "afcghebd", "afcghedb", "afchbdeg", "afchbdge", "afchbedg", "afchbegd", "afchbgde", "afchbged", "afchdbeg", "afchdbge", "afchdebg", "afchdegb", "afchdgbe", "afchdgeb", "afchebdg", "afchebgd", "afchedbg", "afchedgb", "afchegbd", "afchegdb", "afchgbde", "afchgbed", "afchgdbe", "afchgdeb", "afchgebd", "afchgedb", "afdbcegh", "afdbcehg", "afdbcgeh", "afdbcghe", "afdbcheg", "afdbchge", "afdbecgh", "afdbechg", "afdbegch", "afdbeghc", "afdbehcg", "afdbehgc", "afdbgceh", "afdbgche", "afdbgech", "afdbgehc", "afdbghce", "afdbghec", "afdbhceg", "afdbhcge", "afdbhecg", "afdbhegc", "afdbhgce", "afdbhgec", "afdcbegh", "afdcbehg", "afdcbgeh", "afdcbghe", "afdcbheg", "afdcbhge", "afdcebgh", "afdcebhg", "afdcegbh", "afdceghb", "afdcehbg", "afdcehgb", "afdcgbeh", "afdcgbhe", "afdcgebh", "afdcgehb", "afdcghbe", "afdcgheb", "afdchbeg", "afdchbge", "afdchebg", "afdchegb", "afdchgbe", "afdchgeb", "afdebcgh", "afdebchg", "afdebgch", "afdebghc", "afdebhcg", "afdebhgc", "afdecbgh", "afdecbhg", "afdecgbh", "afdecghb", "afdechbg", "afdechgb", "afdegbch", "afdegbhc", "afdegcbh", "afdegchb", "afdeghbc", "afdeghcb", "afdehbcg", "afdehbgc", "afdehcbg", "afdehcgb", "afdehgbc", "afdehgcb", "afdgbceh", "afdgbche", "afdgbech", "afdgbehc", "afdgbhce", "afdgbhec", "afdgcbeh", "afdgcbhe", "afdgcebh", "afdgcehb", "afdgchbe", "afdgcheb", "afdgebch", "afdgebhc", "afdgecbh", "afdgechb", "afdgehbc", "afdgehcb", "afdghbce", "afdghbec", "afdghcbe", "afdghceb", "afdghebc", "afdghecb", "afdhbceg", "afdhbcge", "afdhbecg", "afdhbegc", "afdhbgce", "afdhbgec", "afdhcbeg", "afdhcbge", "afdhcebg", "afdhcegb", "afdhcgbe", "afdhcgeb", "afdhebcg", "afdhebgc", "afdhecbg", "afdhecgb", "afdhegbc", "afdhegcb", "afdhgbce", "afdhgbec", "afdhgcbe", "afdhgceb", "afdhgebc", "afdhgecb", "afebcdgh", "afebcdhg", "afebcgdh", "afebcghd", "afebchdg", "afebchgd", "afebdcgh", "afebdchg", "afebdgch", "afebdghc", "afebdhcg", "afebdhgc", "afebgcdh", "afebgchd", "afebgdch", "afebgdhc", "afebghcd", "afebghdc", "afebhcdg", "afebhcgd", "afebhdcg", "afebhdgc", "afebhgcd", "afebhgdc", "afecbdgh", "afecbdhg", "afecbgdh", "afecbghd", "afecbhdg", "afecbhgd", "afecdbgh", "afecdbhg", "afecdgbh", "afecdghb", "afecdhbg", "afecdhgb", "afecgbdh", "afecgbhd", "afecgdbh", "afecgdhb", "afecghbd", "afecghdb", "afechbdg", "afechbgd", "afechdbg", "afechdgb", "afechgbd", "afechgdb", "afedbcgh", "afedbchg", "afedbgch", "afedbghc", "afedbhcg", "afedbhgc", "afedcbgh", "afedcbhg", "afedcgbh", "afedcghb", "afedchbg", "afedchgb", "afedgbch", "afedgbhc", "afedgcbh", "afedgchb", "afedghbc", "afedghcb", "afedhbcg", "afedhbgc", "afedhcbg", "afedhcgb", "afedhgbc", "afedhgcb", "afegbcdh", "afegbchd", "afegbdch", "afegbdhc", "afegbhcd", "afegbhdc", "afegcbdh", "afegcbhd", "afegcdbh", "afegcdhb", "afegchbd", "afegchdb", "afegdbch", "afegdbhc", "afegdcbh", "afegdchb", "afegdhbc", "afegdhcb", "afeghbcd", "afeghbdc", "afeghcbd", "afeghcdb", "afeghdbc", "afeghdcb", "afehbcdg", "afehbcgd", "afehbdcg", "afehbdgc", "afehbgcd", "afehbgdc", "afehcbdg", "afehcbgd", "afehcdbg", "afehcdgb", "afehcgbd", "afehcgdb", "afehdbcg", "afehdbgc", "afehdcbg", "afehdcgb", "afehdgbc", "afehdgcb", "afehgbcd", "afehgbdc", "afehgcbd", "afehgcdb", "afehgdbc", "afehgdcb", "afgbcdeh", "afgbcdhe", "afgbcedh", "afgbcehd", "afgbchde", "afgbched", "afgbdceh", "afgbdche", "afgbdech", "afgbdehc", "afgbdhce", "afgbdhec", "afgbecdh", "afgbechd", "afgbedch", "afgbedhc", "afgbehcd", "afgbehdc", "afgbhcde", "afgbhced", "afgbhdce", "afgbhdec", "afgbhecd", "afgbhedc", "afgcbdeh", "afgcbdhe", "afgcbedh", "afgcbehd", "afgcbhde", "afgcbhed", "afgcdbeh", "afgcdbhe", "afgcdebh", "afgcdehb", "afgcdhbe", "afgcdheb", "afgcebdh", "afgcebhd", "afgcedbh", "afgcedhb", "afgcehbd", "afgcehdb", "afgchbde", "afgchbed", "afgchdbe", "afgchdeb", "afgchebd", "afgchedb", "afgdbceh", "afgdbche", "afgdbech", "afgdbehc", "afgdbhce", "afgdbhec", "afgdcbeh", "afgdcbhe", "afgdcebh", "afgdcehb", "afgdchbe", "afgdcheb", "afgdebch", "afgdebhc", "afgdecbh", "afgdechb", "afgdehbc", "afgdehcb", "afgdhbce", "afgdhbec", "afgdhcbe", "afgdhceb", "afgdhebc", "afgdhecb", "afgebcdh", "afgebchd", "afgebdch", "afgebdhc", "afgebhcd", "afgebhdc", "afgecbdh", "afgecbhd", "afgecdbh", "afgecdhb", "afgechbd", "afgechdb", "afgedbch", "afgedbhc", "afgedcbh", "afgedchb", "afgedhbc", "afgedhcb", "afgehbcd", "afgehbdc", "afgehcbd", "afgehcdb", "afgehdbc", "afgehdcb", "afghbcde", "afghbced", "afghbdce", "afghbdec", "afghbecd", "afghbedc", "afghcbde", "afghcbed", "afghcdbe", "afghcdeb", "afghcebd", "afghcedb", "afghdbce", "afghdbec", "afghdcbe", "afghdceb", "afghdebc", "afghdecb", "afghebcd", "afghebdc", "afghecbd", "afghecdb", "afghedbc", "afghedcb", "afhbcdeg", "afhbcdge", "afhbcedg", "afhbcegd", "afhbcgde", "afhbcged", "afhbdceg", "afhbdcge", "afhbdecg", "afhbdegc", "afhbdgce", "afhbdgec", "afhbecdg", "afhbecgd", "afhbedcg", "afhbedgc", "afhbegcd", "afhbegdc", "afhbgcde", "afhbgced", "afhbgdce", "afhbgdec", "afhbgecd", "afhbgedc", "afhcbdeg", "afhcbdge", "afhcbedg", "afhcbegd", "afhcbgde", "afhcbged", "afhcdbeg", "afhcdbge", "afhcdebg", "afhcdegb", "afhcdgbe", "afhcdgeb", "afhcebdg", "afhcebgd", "afhcedbg", "afhcedgb", "afhcegbd", "afhcegdb", "afhcgbde", "afhcgbed", "afhcgdbe", "afhcgdeb", "afhcgebd", "afhcgedb", "afhdbceg", "afhdbcge", "afhdbecg", "afhdbegc", "afhdbgce", "afhdbgec", "afhdcbeg", "afhdcbge", "afhdcebg", "afhdcegb", "afhdcgbe", "afhdcgeb", "afhdebcg", "afhdebgc", "afhdecbg", "afhdecgb", "afhdegbc", "afhdegcb", "afhdgbce", "afhdgbec", "afhdgcbe", "afhdgceb", "afhdgebc", "afhdgecb", "afhebcdg", "afhebcgd", "afhebdcg", "afhebdgc", "afhebgcd", "afhebgdc", "afhecbdg", "afhecbgd", "afhecdbg", "afhecdgb", "afhecgbd", "afhecgdb", "afhedbcg", "afhedbgc", "afhedcbg", "afhedcgb", "afhedgbc", "afhedgcb", "afhegbcd", "afhegbdc", "afhegcbd", "afhegcdb", "afhegdbc", "afhegdcb", "afhgbcde", "afhgbced", "afhgbdce", "afhgbdec", "afhgbecd", "afhgbedc", "afhgcbde", "afhgcbed", "afhgcdbe", "afhgcdeb", "afhgcebd", "afhgcedb", "afhgdbce", "afhgdbec", "afhgdcbe", "afhgdceb", "afhgdebc", "afhgdecb", "afhgebcd", "afhgebdc", "afhgecbd", "afhgecdb", "afhgedbc", "afhgedcb", "agbcdefh", "agbcdehf", "agbcdfeh", "agbcdfhe", "agbcdhef", "agbcdhfe", "agbcedfh", "agbcedhf", "agbcefdh", "agbcefhd", "agbcehdf", "agbcehfd", "agbcfdeh", "agbcfdhe", "agbcfedh", "agbcfehd", "agbcfhde", "agbcfhed", "agbchdef", "agbchdfe", "agbchedf", "agbchefd", "agbchfde", "agbchfed", "agbdcefh", "agbdcehf", "agbdcfeh", "agbdcfhe", "agbdchef", "agbdchfe", "agbdecfh", "agbdechf", "agbdefch", "agbdefhc", "agbdehcf", "agbdehfc", "agbdfceh", "agbdfche", "agbdfech", "agbdfehc", "agbdfhce", "agbdfhec", "agbdhcef", "agbdhcfe", "agbdhecf", "agbdhefc", "agbdhfce", "agbdhfec", "agbecdfh", "agbecdhf", "agbecfdh", "agbecfhd", "agbechdf", "agbechfd", "agbedcfh", "agbedchf", "agbedfch", "agbedfhc", "agbedhcf", "agbedhfc", "agbefcdh", "agbefchd", "agbefdch", "agbefdhc", "agbefhcd", "agbefhdc", "agbehcdf", "agbehcfd", "agbehdcf", "agbehdfc", "agbehfcd", "agbehfdc", "agbfcdeh", "agbfcdhe", "agbfcedh", "agbfcehd", "agbfchde", "agbfched", "agbfdceh", "agbfdche", "agbfdech", "agbfdehc", "agbfdhce", "agbfdhec", "agbfecdh", "agbfechd", "agbfedch", "agbfedhc", "agbfehcd", "agbfehdc", "agbfhcde", "agbfhced", "agbfhdce", "agbfhdec", "agbfhecd", "agbfhedc", "agbhcdef", "agbhcdfe", "agbhcedf", "agbhcefd", "agbhcfde", "agbhcfed", "agbhdcef", "agbhdcfe", "agbhdecf", "agbhdefc"]

    hasla = hasla + hasla + hasla + hasla + hasla + hasla

    ce = CharacterEmbeddings(hasla, learningrate=1e-3, epochs=100000, hiddenvectors=4, device="auto", printstep=10000)
