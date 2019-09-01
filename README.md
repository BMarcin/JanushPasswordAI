# JanushPasswordAI

## Nasze środowisko
Character Embeddings wyznaczono na karcie GTX 1060 3GB, jednak algorytm tam napisany jest bardzo zasobo żerny 🙃.
Budowanie modelu znakowego natomiast odywało się na karcie GTX 1080 Ti 11GB.

## Cel projektu
Naszym celem jest zwrócenie uwagi internautów do zwracania uwagi na tworzenie bezpiecznych haseł i ukazanie jakie zagrożenie przy ich zgadywaniu może stanowić sztuczna inteligencja. Chcieliśmy odtworzyć (lub nawet polepszyć) wyniki zaprezentowane w publikacji [PassGAN: A Deep Learning Approach for Password Guessing](https://arxiv.org/pdf/1709.00440.pdf#page=11&zoom=100,0,358). Dodatkowo chcemy zachęcić jak największe grono osób do korzystania z menedżerów haseł, by kolejne wycieki haseł odbiły się na każdym w jak najmniejszym stopniu. Kradzież tożsamości to codzienność, która dotyczy wielu z nas.

## Pliki modelu:
- Notebooks/EvenBetterLSTM
- Notebooks/ModelEvaluation

## Opis wybranej publikacji
W publikacji [PassGAN: A Deep Learning Approach for Password Guessing](https://arxiv.org/pdf/1709.00440.pdf#page=11&zoom=100,0,358) opracowanej przez badaczy instytutu Stevens Institute of Technology w New Jersey oraz New York Institute of Technology przedstawiono wysoce skuteczną opartą o sieć GAN (Generative Adversarial Network) metodę odgadywania haseł - PassGAN. Przeprowadzone eksperymenty wykazują lepszą skuteczność nowej metody w stosunku do najnowocześniejszych, opartych na regułach (np. konkatenacja słów oraz leet speak) narzędzi z dziedziny machine learning do odgadywania haseł, takich jak HashCat i JTR. Należy jednak nadmienić, że dobre rezultaty zostały osiągnięte bez wiedzy o samych hasłach jak i ich strukturze.
Bazując na dwóch zbiorach haseł pochodzących z wycieków RockYou oraz LinkedIn udało im się osiągnąć skuteczność dopasowań unikatowych haseł kolejno na poziomie ~44% i ~24%. Mimo niezbyt zachęcających wyników spora liczba wygenerowanych  nieadekwatnych haseł przypominała te oryginalnie wytworzone przez człowieka, które potencjalnie mogły odpowiadać rzeczywistym hasłom nie branym pod uwagę w eksperymentach. Dodatkowo PassGan w połączeniu HashCat był w stanie poprawnie odgadywać między 51% a 73% więcej haseł niż sam HashCat.
W założeniu PassGAN  wytrenowana sieć autonomicznie określa cechy hasła oraz jego strukturę, by następnie na podstawie nabytej wiedzy wygenerować nowe próbki, które odpowiadają dystrybucji i do złudzenia przypominają hasła rzeczywiste.
PassGAN wykorzystuje sieci konwolucyjne.

## Dane
Podstawą naszego zbioru danych będą bazy danych z wycieków polskich serwisów. Chcielibyśmy zebrać około 1 miliona realnych rekordów zawierających loginy oraz hasła użytkowników polskich serwisów. W tym celu będziemy musieli wyszukać konta pozakładane przez boty i odfiltrować je od pozostałych, by nie zakłócały one “ludzkiego” schematu haseł.
Głównym problemem związanym z danymi będzie konieczność ich odhaszowania. W związku z tym nie będziemy w stanie wykorzystać w 100% każdego ze znalezionych zbiorów, gdyż odhaszowanie niektórych haseł może zająć bardzo dużo czasu. Hasła takie bardzo często mogą być generowane przez menedżery haseł, co by tylko zaburzało humanoidalną ideę haseł.
Do odhaszowania użyjemy programu HashCat wykorzystującego tryb hybrydowy (tryb 7), słownik bazujący na książce “A Dictionary of Fabulous Beasts” oraz część masek zdefiniowanych przez twórców HashCat’a “rockyou-6-864000”.
Posiadane loginy chcemy wykorzystać do przedstawienia statystyk związanych z hasłami jakie są przechowywane w polskich serwisach, by móc również zaprezentować jak bardzo jest to bagatelizowana kwestia np. poprzez najczęściej powtarzające się loginy oraz hasła.
Planujemy cały zbiór podzielić na dwie części - zbiór uczący (90%) oraz zbiór testowy (10%), który umożliwi nam sprawdzenie i porównanie wyników.


## Wykorzystywane metody uczenia maszynowego
Zdecydowaliśmy się wzorować na modelach językowych i wykorzystaliśmy rekurencyjne sieci neuronowe oparte o komórki GRU. 

![Przygotowany przez nas model](https://i.imgur.com/ytNEXZc.png)

## W jaki sposób przygotowany przez nas model stanowi zagrożenie dla haseł?
Sieć może generować ciągi znaków na podstawie wyuczonych reguł - prawdopodobieństw występowania po sobie kolejnych znaków. Tego typu rozwiązania nie oferuje żaden z obecnych crackerów haseł. Najlepszą opcją według nas jest wygenerowanie X haseł za pomocą wyuczonego modelu i następnie udanie się z tak przygotowanym słownikiem do np. Hashcat'a. 

## Przykładowe hasła generowane przez model
Ania1969
vaceria1
!romek18
8732621
Yosty12344
-eror71
Bitek123
&wothald
jorek1996
)onek1985
Karunia1
qwociszz
Dalia1977
Paponanya
Onaki1988
\%eltek171
lena1955
ura12334
Rebi1983
=latek1
Cania1986
Doranta1
wydus1988
\*vastankey
=borek176
$rike1977
usosia1
Ouki1969
Qrassa200
5021000
lita9000
5446486665
łlrota123
hancer12
jestela
Qarek22
02054311
_uinderu77
,amila11
ilapolina
6512120
Fisces666
Fabarek
KEGELA
unia1979
Xasia1982
Kania2009
Para1971
olka1234
qwerdann
ONWA12344
0010051
Xukasia1
Ania611
ONAN1967
kopradan
Zobiniek
Barta13
Rustat1
Disio1985

## Autorzy
[Marcin Borzymowski](https://github.com/BMarcin)
[Adam Chrzanowski](https://github.com/chradam)
[Aleksandra Marzec](https://github.com/AleksMarzec)


## Bibliografia
+ [https://arxiv.org/pdf/1709.00440.pdf#page=11&zoom=100,0,358](https://arxiv.org/pdf/1709.00440.pdf#page=11&zoom=100,0,358)
+ [https://www.darkreading.com/analytics/passgan-password-cracking-using-machine-learning/d/d-id/1329964](https://www.darkreading.com/analytics/passgan-password-cracking-using-machine-learning/d/d-id/1329964)
+ [https://arxiv.org/pdf/1406.2661.pdf](https://arxiv.org/pdf/1406.2661.pdf)
