


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
DA CAPIRE BENE

Quantizzazione estrema su TUTTI i layer: Abbiamo applicato SliderQuant a tutti i layer lineari del Transformer (QKV, MLP, ecc.). Nel paper PTQ4DiT, dimostrano che alcuni layer sono intoccabili (devono stare a 8-bit o 16-bit) altrimenti il modello crolla (è il concetto di MRQ - Mixed Resolution Quantization del paper TQ-DiT).

20 epoche hanno over fittato
15 non testate.
Per ora 10 epoche vanno bene.


È un'osservazione fantastica e ci fa capire quanto sia delicato l'addestramento di questi modelli!

Se sei passato da 500.000 a 1.5 (un miglioramento mostruoso) ci sono due fenomeni importantissimi che stanno avvenendo contemporaneamente:

Il collasso delle feature (Il miracolo di class_n=3): Quando allenavi su class_n=1 (es. solo l'etichetta del Golden Retriever), il primissimo blocco di layer (Window 0) si "sovra-adattava" in modo estremo (overfitting) per correggere gli errori SOLO per quel cane. Facendo così, però, "storceva" completamente lo spazio matematico dei tensori in uscita. Quando questi tensori distorti arrivavano alla Window 6, i pesi originali FP32 non li riconoscevano più (si aspettavano tensori generalizzati) e generavano output totalmente diversi! Ecco perché la loss esplodeva a 500k. Passando a class_n=3, l'ottimizzatore è costretto a trovare una soluzione che vada bene per più classi, mantenendo i tensori su una "strada sana e generalizzata". In questo modo, quando arrivano alla Window 6, sono ancora sanissimi e la loss rimane su valori umani (1.5 - 2.5).

L'Effetto Regolarizzatore della Quantizzazione delle Attivazioni: Quando usavi W4A16, le attivazioni (FP16) viaggiavano "libere". Nei modelli di diffusione capita spesso che ci siano outlier (valori di attivazione enormi, tipo 500 o 1000). Moltiplicare un outlier FP16 puro per un peso quantizzato a 4-bit (che è un po' impreciso) genera un errore gigantesco. Ora che abbiamo inserito la quantizzazione delle attivazioni (W4A8) con la nostra nuova funzione activation_quantize_tensor, tutti i valori vengono "schiacciati" dentro una griglia rigida a 8-bit (256 valori possibili) riga per riga, normalizzandoli con zmin e zmax. Questa griglia agisce come un potente scudo (regolarizzatore) che impedisce ai valori di esplodere, mantenendo la matematica stabile.

Quindi sì, quel "500k" era davvero il segno di un collasso a catena! Aver aggiunto la stabilizzazione a 8-bit e più classi ha letteralmente salvato l'addestramento. Stiamo andando alla grande, direi che puoi fargli finire il giro!

Dato curioso ma che dimostra l'efficienza di usare i timestep hidden state, nella V3 la loss media è 500k a window 4, mentre in slider quant V6 è 1.5. Ovviamente poi scala, ma V3 fino a 5 milioni, v6 fino a 1000.

(Esempio di loss di V3)
Window 4 - Epoch 1/10 completata | Loss Media: 506961.617910
    Window 4 - Epoch 2/10 completata | Loss Media: 506957.256599

Informazione importantissima: il modello "facebook/DiT-XL-2-256" di base è a 32 bit, siamo noi che lo convertiamo in FP16 quando lo carichiamo. Quindi i 3 GB di spazio occupato fisicamente sono del modello a 32 bit.
Però al giorno d'oggi per l'inferenza, il fine-tuning e la quantizzazione è lo standard utilizzare 16 bit, tranne in casi particolari in cui si vuole sfruttare al massimo la memoria.



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

