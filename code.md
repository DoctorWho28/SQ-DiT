


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



NOI SALVIAMO I MODELLI IN QUESTO MODO NELLA CARTELLA FACEBOOK PER ESEMPIO:
    -DiT-XL-2-256-w2a2-2
    -DiT-XL-2-256-w2a2-3

-- La parte 3 segue questa logica: mentre si traina il modello quantizzato l'input del layer 0 per ogni bucket è l'output del modello originale per il bucket precedente. La propagazione dell'errore (e la conseguente prova di correzione) viene resettata ad ogni bucket perché non sarebbe possibile quando si sta trainando il layer 0 sapere quale errore si è propagato dal layer finale nel bucket precedente.


---------
### Analisi del Comportamento del Modello e Risoluzione del Collasso

**1. Quantizzazione Estrema e Layer Sensibili**
Abbiamo applicato `SliderQuant` a *tutti* i layer lineari del Transformer (QKV, MLP, ecc.). Come dimostrato dalla letteratura (es. paper PTQ4DiT e TQ-DiT), la quantizzazione W4A8 uniforme può essere rischiosa: alcuni layer sono estremamente sensibili e andrebbero mantenuti a precisione maggiore (es. 8-bit o 16-bit) per prevenire il collasso del modello (concetto di MRQ - *Mixed Resolution Quantization*).

**2. Tuning delle Epoche**
- **20 epoche:** Rischio di overfitting (il modello si adatta troppo ai dati di addestramento perdendo generalizzazione).
- **10-15 epoche:** Rappresentano attualmente il miglior compromesso tra ottimizzazione e generalizzazione (stiamo testando anche 30 epoche, compensate però da un alto numero di classi).

**3. Risoluzione dell'Esplosione della Loss (Da 500k a 1.5)**
Nelle versioni iniziali (es. V3), la Loss esplodeva a valori anomali (es. 500.000 alla Window 4). Questo fenomeno di "collasso a catena" è stato mitigato tramite due interventi chiave:

- **Prevenzione del Collasso delle Feature (Aumento delle Classi):** 
  Allenare su una singola classe (`class_n=1`) portava il primissimo blocco di layer (Window 0) in forte overfitting. Questo distorceva irreparabilmente lo spazio matematico dei tensori in uscita. Arrivati ai layer più profondi (es. Window 6), i pesi originali FP32 non riconoscevano più l'input, facendo esplodere la Loss. Aumentando le classi (`class_n=3` o `20`), l'ottimizzatore è costretto a generalizzare. Le distribuzioni dei tensori si mantengono "sane" lungo tutta la rete, stabilizzando la Loss su valori normali (1.5 - 2.5).

- **Effetto Regolarizzatore della Quantizzazione delle Attivazioni (W4A8 vs W4A16):** 
  I modelli di diffusione presentano spesso outlier di attivazione estremi. Mantenendo le attivazioni "libere" in FP16, questi outlier venivano moltiplicati per i pesi a 4-bit (imprecisi per natura), amplificando a dismisura l'errore. Quantizzando anche le attivazioni a 8-bit tramite la funzione `activation_quantize_tensor`, i valori vengono compressi in una griglia discreta (256 valori possibili) e normalizzati. Questa "gabbia a 8-bit" funge da potente scudo regolarizzatore, bloccando l'errore matematico a catena causato dagli outlier FP16.


### Note per il Report / Benchmark FID
- **Numeri Ufficiali:** Prendere sempre i punteggi FID dei paper originali (es. PTQ4DiT, Q-Diffusion), ignorando quelli calcolati e riportati dai loro "competitor".
- **La Regola del Delta ($\Delta$FID):** Poiché noi valutiamo a 20 step (per efficienza) e loro a 250, i valori FID assoluti non coincidono. Il vero confronto va fatto sulla *perdita* (Delta) rispetto alla baseline FP16 (es: "Il metodo X degrada il FID di +2.0 punti, il nostro di soli +1.5 punti a parità di step").
- **Onestà Intellettuale (Evaluation Setup):** Dichiarare sempre apertamente nel report: *"A causa dei costi computazionali, tutti i modelli sono valutati a 20 step di inferenza. Per compensare l'impossibilità di confrontare i FID assoluti con i paper ufficiali (250 step), il benchmark si basa sulla degradazione relativa ($\Delta$FID) calcolata sulla nostra baseline FP16 locale, garantendo così una metrica di robustezza equa e affidabile."*
- **Miglioramento del FID a bassi step (L'Effetto Regolarizzazione):** Come supportato dalla letteratura recente (es. Tabella 2 in PTQD, *He et al., 2023*), in scenari di inferenza rapida a 20 step, l'introduzione della quantizzazione agisce come regolarizzatore del rumore spaziale. Questo compensa la deriva della distribuzione del modello, permettendo al modello W4A8 di superare matematicamente le prestazioni della controparte Full Precision (ottenendo un FID più basso). Lo stesso fenomeno di "sorpasso" della baseline è documentato anche in Q-Diffusion (*Li et al., 2023*, Tabella 3).
- La maggior parte dei paper usa 10k immagini per il FiD e il resto.

-TODO:
    - Controllare come gestire il caso in cui i bit siano 16
    - Settare l'assert per quanto riguarda i bit.
    - Capire come gestire il running della quantizzazione e della generazione delle immagini
    - Decidere i parametri da utilizzare per le varie fasi e quantizzazioni


Ordine delle cose da runnare:
    - quantizer (--model, --config)
    - gen_fid_images
    - calculate_metrics

Per creare solo un'immagine:
    - inference

