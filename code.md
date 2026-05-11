


Passi del codice:
1. Calcolare i bucket:
    - Runnare il modello in base ai timestamp
    - Confrontare gli output del layer per suddividere i timestamp in gruppi
    - Ritornare una lista di timestamp che sono in bucket diversi

2. Calcolare i layer shallow e deep:
    - Quantizzare un layer alla volta
    - Runnare per tutti i bucket
    - Calcolare l'MSE tra i risultati del layer quantizzato e quello originale
    - Dividere in layer nei 3 gruppi: shallow, intermediate e deep

3. Applicare la quantizzazione fisica (SliderQuant):
    - Calcolare gli input del modello originale per ogni classe e per ogni bucket
    - Calcolare le window dei layer
    - Per ogni window:
        - Quantizzare tutti i layer della window (prima solo in percentuale con gamma e poi tutti)
        - Per x epoche (probabilmente 1):
            - Per ogni input (class_id):
                - Per ogni bucket:
                    - Runnare con input quantizzato precedente (l'output del layer precedente al primo della window attuale dello stesso bucket)
            - Calcolare l'MSE tra i risultati della window quantizzata e quello originala
            - Optimizer AdamW

4. Applicare la quantizzazione delle attivazioni



-- La parte 3 segue questa logica: mentre si traina il modello quantizzato l'input del layer 0 per ogni bucket è l'output del modello originale per il bucket precedente. La propagazione dell'errore (e la conseguente prova di correzione) viene resettata ad ogni bucket perché non sarebbe possibile quando si sta trainando il layer 0 sapere quale errore si è propagato dal layer finale nel bucket precedente.