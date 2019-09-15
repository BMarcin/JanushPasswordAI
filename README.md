# JanushPasswordAI

## Opis
JanushPasswordAI to sztuczna inteligencja, która wykorzystuje rekurencyjne sieci neuronowe do generowania haseł podobnych do tych, które tworzą ludzie. Główną myślą przewodnią było ukazanie ludziom, jak tworzenie słabych haseł i połączenie algorytmów sztucznej inteligencji może zagrozić bezpieczeństwu ich wirtualnego życia. Możemy dumnie stwierdzić, że cel został osiągnięty, a efekty naszych eksperymentów przedstawiamy w tym repozytorium 😎.

## W jaki sposób przygotowany przez nas model stanowi zagrożenie dla haseł?
Sieć może generować ciągi znaków na podstawie wyuczonych reguł - prawdopodobieństw występowania po sobie kolejnych znaków. Tego typu rozwiązania nie oferuje żaden z obecnych crackerów haseł. Najlepszą opcją według nas jest wygenerowanie X haseł za pomocą wyuczonego modelu i następnie udanie się z tak przygotowanym słownikiem do np. Hashcat'a. 

## Dane
Udało nam się zebrać hasła z kilku wycieków danych polskich serwisów, celowaliśmy głównie w hasła zahaszowane skrótem MD5, gdyż jest łatwy w odtworzeniu.
Głównym problemem związanym z danymi będzie konieczność ich odhaszowania. Postawiliśmy na własnoręczne odhaszowanie, by mieć pewność, że hasła są prawdziwe i były wykorzystywane przez ludzi. 
Do odhaszowania użyjemy programu HashCat wykorzystującego tryb hybrydowy (tryb 7), słownik bazujący na książce “A Dictionary of Fabulous Beasts” oraz część masek zdefiniowanych przez twórców HashCat’a “rockyou-6-864000”.
Ostatecznie udało nam się uzyskać 178 664 hasła, które podzieliliśmy na dwa zbiory - uczący oraz walidacyjny - odpowiednio 95%(169 730) oraz 5% (8934).

## Walidacja
Zbiór walidacyjny służyć ma do sprawdzenia, czy przygotowany przez nas model będzie w stanie stworzyć ludzkie hasło oraz wyznaczenia schematów (litery oraz cyfry) haseł i porównania ich ze schematami tworzonymi przez model.
Porównywanie haseł utworzonych przez nasze AI z hasłami ze zbioru walidacyjnego nie ma większego sensu, gdyż nawet na etapie uczenia mogliśmy spotkać hasła o wspólnym początku, a innych końcówkach. Model nie ma żadnych podstaw do wnioskowania, które zakończenie należy wykorzystać w danym momencie i to samo tyczy się etapu ewaluacji. Taki stan rzeczy doprowadza do sytuacji w której w zbiorze walidacyjnym będziemy mieć np. hasło "kicia08", a nasz model wygeneruje hasło "kicia09". Występowanie obu końcówek jest równie prawdopodobne (prawdopodobnie 😆), a metoda porównań 1:1 nie jest w stanie tego przedstawić.
Metoda schematowa jest w stanie to opisać i dała nam bardzo dobre wyniki. Sieć wygenerowała 322 hasła (hasła, które się powtarzały były liczbowe oraz głownie składały się z liter na początku i cyfr na końcu), które idealnie odwzorowywały hasła ze zbioru walidacyjnego (to około 3.60% pokrycia). Sprawdzając natomiast strukturę wygenerowanych haseł (metoda schematowa) uzyskaliśmy **88.40%** pokrycia (do tego wyniku nie wliczają się hasła, które powtórzyły się 1:1 w zbiorach)! 


## Architektura modelu
Zdecydowaliśmy się wzorować na modelach językowych i wykorzystaliśmy rekurencyjne sieci neuronowe oparte o dwukierunkowe komórki GRU. Model posiada 15 takich komórek oraz rozmiar *hidden* = 40. Ostatecznie otrzymaliśmy model, który posiada 456 990 parametrów podlegających uczeniu.

![Przygotowany przez nas model](https://i.imgur.com/ytNEXZc.png)
Model opiera się o prawdopodobieństwo warunkowe występowania po sobie kolejnych znaków pod warunkiem poprzednich, dlatego w celu uzyskiwania zróżnicowanych haseł (podczas generowania) wykorzystujemy próbkowanie każdego kolejnego znaku przez wielomianowy rozkład prawdopodobieństwa.

## Nasze środowisko
Budowanie modelu  odywało się na karcie GTX 1080 Ti 11GB z wykorzystaniem frameworka pyTorch.

### Hiperparametry
```
lr_start = 1e-3
lr_end = 1e-8
epochs = 50
batch_size = 1500
```
Jako funkcję straty wykorzystaliśmy *CrossEntropyLoss* oraz zdecydowaliśmy się na wykorzystanie zmiennej wartości *learning rate* zgodnie z *CosineAnnealing*.


## Przykładowe hasła generowane przez model
Hasła zostały wybrane z losowego ciągu wygenerowanych haseł. Autorzy nie ingerowali w nie celem sztucznej poprawy wyników. Plik z wygenerowanymi hasłami jest dostępny w repozytorium 🐱‍👤.

harsat19
tpjko252
16091982
olieketolona
bezos121
zcyczu11222
123vyxzix
kazek655
tomailousza
mika61122
erzeczka
concetas12
1tkoramanek
vigenter1
majakaba11
anetaangsm
8841502199
flompbspj
barbika19
emi48963
tata1112
rafi77ka
bobarada11
PUSIA1279
frakkoporn
olabolakd
heag123446
ejalo6343023
kokaladka
anka2010
char82
bpmrolinek
Matrissdnl
exzd011
marys2011
n57853961
ae2625379
akimon1989
adidanrinitise
ratiskon0010
manusia1111
iza1999
pinka1004
fiesta13
tytos1125
110169
pasiker54320
piller1234

## Co znajdziesz w repo?
Repo zawiera całość efektów naszej pracy wraz z przeuczonym modelem. Jeżeli chcesz na własnym sprzęcie przetestować model, to wystarczy "wyklikać" odpowiednie Notebooki. Przygotowaliśmy osobny notebook dla uczenia modelu, podziału datasetu, usuwania duplikatów (tak, modelowi się to zdarza 😆), ewaluacji (generowania hasełek) oraz walidacji, czyli wykorzystania naszych metod do reprezentacji pokryć zbiorów. W repozytorium postanowiliśmy nie umieszczać odszyfrowanych haseł, by choć w jakiejś części pozostać etycznymi 🐱‍🏍.

## Autorzy
[Marcin Borzymowski](https://github.com/BMarcin)
[Adam Chrzanowski](https://github.com/chradam)
[Aleksandra Marzec](https://github.com/AleksMarzec)

## Podziękowania
Chcemy serdecznie podziękować firmie [theBlue.ai](https://theblue.ai), która udostępniła nam nielimitowany dostęp do własnych zasobów obliczeniowych, by móc trenować wiele modeli jednocześnie, co w znaczący sposób przyspieszyło prace nad projektem i umożliwiło przygotowanie działającego modelu, a pracownikom za wskazanie drogi jaką były sieci odwzorowujące sekwencje

## Bibliografia
+ [https://arxiv.org/pdf/1709.00440.pdf#page=11&zoom=100,0,358](https://arxiv.org/pdf/1709.00440.pdf#page=11&zoom=100,0,358)
+ [https://www.darkreading.com/analytics/passgan-password-cracking-using-machine-learning/d/d-id/1329964](https://www.darkreading.com/analytics/passgan-password-cracking-using-machine-learning/d/d-id/1329964)
+ [https://arxiv.org/pdf/1406.2661.pdf](https://arxiv.org/pdf/1406.2661.pdf)
