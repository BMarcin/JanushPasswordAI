# JanushPasswordGAN

## Ważne info
Tak wiem, pyTorch oferuje embeddings jako gotową funkcję, jednak chciałem dowiedzieć się dokładnie jak to działa.

## Instalacja venv
```
conda env create -f environment.yaml
```

## Nasze środowisko
Klasa pythonowa była testowana na zbiorze danych wejściowych *password3.txt*. Dane zapełniły 3GB RAMu karty graficznej(GTX 1060) i obciążyły ją na około 40%.

## Cel projektu
Naszym celem będzie utworzenie sztucznej sieci neuronowej typu GAN odpowiedzialnej za tworzenie haseł, które mógłby wymyślić człowiek. Chcielibyśmy odtworzyć (lub nawet polepszyć) wyniki zaprezentowane w publikacji [PassGAN: A Deep Learning Approach for Password Guessing](https://arxiv.org/pdf/1709.00440.pdf#page=11&zoom=100,0,358) oraz zobrazować zagrożenia jakie niosą ze sobą hasła. Osiągnięte przez nas wyniki mogą posłużyć do stworzenia nowych słowników haseł oraz tablic tęczowych dla polskich wersji językowych haseł. Dodatkowo chcemy zachęcić jak największe grono osób do korzystania z menedżerów haseł, by kolejne wycieki haseł odbiły się na każdym w jak najmniejszym stopniu. Kradzież tożsamości to codzienność, która dotyczy wielu z nas.

## Postęp prac
+ Przygotowano wstępny schemat dla CharacterEmbeddings (obecnie jest do poprawy)

## Opis wybranej publikacji
W publikacji [PassGAN: A Deep Learning Approach for Password Guessing](https://arxiv.org/pdf/1709.00440.pdf#page=11&zoom=100,0,358) opracowanej przez badaczy instytutu Stevens Institute of Technology w New Jersey oraz New York Institute of Technology przedstawiono wysoce skuteczną opartą o sieć GAN (Generative Adversarial Network) metodę odgadywania haseł - PassGAN. Przeprowadzone eksperymenty wykazują lepszą skuteczność nowej metody w stosunku do najnowocześniejszych, opartych na regułach (np. konkatenacja słów oraz leet speak) narzędzi z dziedziny machine learning do odgadywania haseł, takich jak HashCat i JTR. Należy jednak nadmienić, że dobre rezultaty zostały osiągnięte bez wiedzy o samych hasłach jak i ich strukturze.
Bazując na dwóch zbiorach haseł pochodzących z wycieków RockYou oraz LinkedIn udało im się osiągnąć skuteczność dopasowań unikatowych haseł kolejno na poziomie ~44% i ~24%. Mimo niezbyt zachęcających wyników spora liczba wygenerowanych  nieadekwatnych haseł przypominała te oryginalnie wytworzone przez człowieka, które potencjalnie mogły odpowiadać rzeczywistym hasłom nie branym pod uwagę w eksperymentach. Dodatkowo PassGan w połączeniu HashCat był w stanie poprawnie odgadywać między 51% a 73% więcej haseł niż sam HashCat.
W założeniu PassGAN  wytrenowana sieć autonomicznie określa cechy hasła oraz jego strukturę, by następnie na podstawie nabytej wiedzy wygenerować nowe próbki, które odpowiadają dystrybucji i do złudzenia przypominają hasła rzeczywiste.

## Dane
Podstawą naszego zbioru danych będą bazy danych z wycieków polskich serwisów. Chcielibyśmy zebrać około 1 miliona realnych rekordów zawierających loginy oraz hasła użytkowników polskich serwisów. W tym celu będziemy musieli wyszukać konta pozakładane przez boty i odfiltrować je od pozostałych, by nie zakłócały one “ludzkiego” schematu haseł.
Głównym problemem związanym z danymi będzie konieczność ich odhaszowania. W związku z tym nie będziemy w stanie wykorzystać w 100% każdego ze znalezionych zbiorów, gdyż odhaszowanie niektórych haseł może zająć bardzo dużo czasu. Hasła takie bardzo często mogą być generowane przez menedżery haseł, co by tylko zaburzało humanoidalną ideę haseł.
Do odhaszowania użyjemy programu HashCat wykorzystującego tryb hybrydowy (tryb 7), słownik bazujący na książce “A Dictionary of Fabulous Beasts” oraz część masek zdefiniowanych przez twórców HashCat’a “rockyou-6-864000”.
Posiadane loginy chcemy wykorzystać do przedstawienia statystyk związanych z hasłami jakie są przechowywane w polskich serwisach, by móc również zaprezentować jak bardzo jest to bagatelizowana kwestia np. poprzez najczęściej powtarzające się loginy oraz hasła.
Planujemy cały zbiór podzielić na dwie części - zbiór uczący (90%) oraz zbiór testowy (10%), który umożliwi nam sprawdzenie i porównanie wyników.


## Wykorzystywane metody uczenia maszynowego
Nasz projekt będzie korzystać z prostej sieci w celu utworzenia character embeddings, by każde hasło miało swoją reprezentację liczbową oraz z sieci GAN.
Sieć GAN służyć będzie do generowania haseł przypominających do złudzenia te, które posłużyły do wytrenowania modelu sieci. 
Składa się ona z dwóch głębokich sieci **generatora (G)** i  **dyskryminatora (D)**. Generator jest odpowiedzialny za wygenerowanie kandydatów a dyskryminator ocenia ich stopień wiarygodności / odwzorowania. 
Proces kształtowania dystrybucji **pg** na zbiorze x przebiega tak, że w pierwszej kolejności na wejście generatora **G** podawany jest szum **pz(z)** zawierający utajone cechy, na podstawie których reprezentuje się mapowanie w przestrzeni danych jako **G(z; teta_g)**, gdzie **G** jest funkcją różniczkowalną reprezentowaną przez wielowarstwowy perceptron z parametrami **teta_g**. Dyskryminator jako drugi wielowarstwowy perceptron **D(x; teta_d)** generuje pojedynczy skalar.
**D(x)** reprezentuje prawdopodobieństwo, że **x** pochodzi od rzeczywistych danych niż z wygenerowanej sztucznie dystrybucji **pg**. Uczymy **D** maksymalizacji prawdopodobieństwa przypisania właściwej etykiety przykładom treningowym i próbkom pozyskanym od generatora. Jednocześnie uczymy generator minimalizować **log(1 - D(G(z)))**.

![Struktura sieci GAN](https://i.imgur.com/uN3wnji.png)

Innymi słowy D oraz G grają w dwuosobową grę minimax z funkcją wartości V(G, D).

![Wykres uczenia sieci](https://i.imgur.com/cB5xo3r.png)

## Autorzy
[Adam Chrzanowski](https://github.com/chradam)
[Aleksandra Marzec](https://github.com/AleksMarzec)
[Marcin Borzymowski](https://github.com/BMarcin)


## Bibliografia
+ [https://arxiv.org/pdf/1709.00440.pdf#page=11&zoom=100,0,358](https://arxiv.org/pdf/1709.00440.pdf#page=11&zoom=100,0,358)
+ [https://www.darkreading.com/analytics/passgan-password-cracking-using-machine-learning/d/d-id/1329964](https://www.darkreading.com/analytics/passgan-password-cracking-using-machine-learning/d/d-id/1329964)
+ [https://en.wikipedia.org/wiki/Generative_adversarial_network](https://en.wikipedia.org/wiki/Generative_adversarial_network)
+ [https://arxiv.org/pdf/1406.2661.pdf](https://arxiv.org/pdf/1406.2661.pdf)
