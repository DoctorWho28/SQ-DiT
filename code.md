


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

TO-DO:
    -Capire perchè calc_original_outputs non funziona e al timestep 680 layer 11 restituisce nan come valori. (risolto credo)
    -Capire come salvare in modo migliore i latent originali di calc_original_outputs (prob torch.save) 

La cartella temporanea è unica, quindi il resume è solo dell'ultima quantizzazione iniziata.

NOI SALVIAMO I MODELLI IN QUESTO MODO NELLA CARTELLA FACEBOOK PER ESEMPIO:
    -DiT-XL-2-256-w2a2_2
    -DiT-XL-2-256-w2a2_3

-- La parte 3 segue questa logica: mentre si traina il modello quantizzato l'input del layer 0 per ogni bucket è l'output del modello originale per il bucket precedente. La propagazione dell'errore (e la conseguente prova di correzione) viene resettata ad ogni bucket perché non sarebbe possibile quando si sta trainando il layer 0 sapere quale errore si è propagato dal layer finale nel bucket precedente.


---------
DA CAPIRE BENE

Quantizzazione estrema su TUTTI i layer: Abbiamo applicato SliderQuant a tutti i layer lineari del Transformer (QKV, MLP, ecc.). Nel paper PTQ4DiT, dimostrano che alcuni layer sono intoccabili (devono stare a 8-bit o 16-bit) altrimenti il modello crolla (è il concetto di MRQ - Mixed Resolution Quantization del paper TQ-DiT).

20 epoche hanno over fittato
15 non testate.
Per ora 10 epoche vanno bene.