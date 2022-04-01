# Soluție hackathon

## Descriere soluție
Am folosit modelul de NER pentru limba română disponibil în biblioteca huggingface, poate fi găsit la link-ul următor: https://huggingface.co/dumitrescustefan/bert-base-romanian-ner.
Am folosit ponderile pre-antrenate, dar am adăugat un strat liniar de 16 neuroni la final, deoarece modelul de pe huggingface funcționează cu 31 de clase, fiind bazat pe un corpus etichetat
în stil BIO2, și am antrenat toate bias-urile modelului.

Ca pre-procesări, am înlocuit orice posibile caractere cu sedilă cu varianta lor cu diacritice, după cum sugereaza Ștefan Dumitrescu, autorul modelului, și am mai înlocuit
câteva caractere pe care tokenizer-ul modelului le elimina din lista de tokeni cu niște denumiri unice. Mai exact, am înlocuit '\n' cu "NTOK", ' ' și '\xao' cu "WS". Deși
la prima vedere poate nu pare o idee foarte bună, am reușit să obținem o performanță de 76% în final cu strategia aceasta de pre-procesare.

Am făcut și niște post-procesări la datele de testare, deoarece tokenizer-ul adaugă tokeni în plus, iar pe alții îi sparge în mai mulți tokeni. Tokenii care nu au corespondent
în tokenii inițiali, cum ar fi '[CLS]' sau '[SEP]' i-am ignorat complet, iar pentru tokenii care corespund unui tokeni din lista originală, am scris în fișierul de submisie
doar primul token dintre ei întâlnit în listă, în ideea că, deoarece am redus clasele la 16, ar trebui ca modelul să învețe să dea aceeași clasă tuturor părților unui
cuvânt.

## Biblioteci utilizate
Lista completă este în fișierul ```requirements.txt```

## Rulare soluție
Este suficientă rularea pe rând a cell-urilor din notebook, cu mențiunea că unele trebuie rulate doar în cazul în care este folosit collab. De asemenea, dacă este rulată
local soluția, atunci datele de testare și antrenare ar trebui să fie la același file level cu notebook-ul.

## Bibliografie
Model folosit: https://huggingface.co/dumitrescustefan/bert-base-romanian-ner (accesat pe 27.03.2022)

Codul de antrenare și evaluare al modelului este bazat pe workshop-ul susținut de Andrei Manolache: https://github.com/Nitro-Language-Processing/Workshops/blob/main/Transfer%20Learning%20and%20Transformers/tutorial_transfer.ipynb (accesat pe 27.03.2022)
