import os
import json
import re
from pathlib import Path


def analizza_stagnazione_hof(cartella_input, soglia_stagnazione=3):
    """
    Analizza i file JSON per rilevare quando la Hall of Fame ristagna.

    :param cartella_input: Percorso della cartella contenente i file JSON.
    :param soglia_stagnazione: Numero minimo di generazioni consecutive con la stessa HoF
                               per considerare la situazione un "ristagno".
    """
    percorso_cartella = Path(cartella_input)

    # 1. Trova tutti i file JSON
    file_json = list(percorso_cartella.glob("*.json"))

    if not file_json:
        print(f"Nessun file JSON trovato nella cartella: {cartella_input}")
        return

    # 2. Funzione per estrarre la generazione dal nome del file (es: ..._gen191.json -> 191)
    def estrai_generazione(filepath):
        match = re.search(r"_gen(\d+)\.json$", filepath.name)
        if match:
            return int(match.group(1))
        return -1  # Se non trova il pattern, lo mette all'inizio

    # Ordina i file per generazione
    file_json.sort(key=estrai_generazione)

    hof_precedente = set()
    contatore_stagnazione = 0
    inizio_stagnazione = None

    print(f"Inizio analisi su {len(file_json)} file...\n")

    # 3. Itera sui file ordinati
    for percorso_file in file_json:
        gen_corrente = estrai_generazione(percorso_file)

        # Ignora i file di cui non capiamo la generazione
        if gen_corrente == -1:
            continue

        # Leggi il JSON
        try:
            with open(percorso_file, 'r', encoding='utf-8') as f:
                dati = json.load(f)
                # Gestisce sia il caso in cui il JSON sia una lista [ {..} ] che un dict {..}
                sezione_dati = dati[0] if isinstance(dati, list) else dati

                # Estrai la Hall of Fame (creiamo un SET con le stringhe degli individui)
                hof_lista = sezione_dati.get("hall_of_fame", [])
                hof_corrente = frozenset([individuo["individual"] for individuo in hof_lista])

        except Exception as e:
            print(f"Errore nella lettura del file {percorso_file.name}: {e}")
            continue

        # 4. Controllo Stagnazione
        if hof_corrente == hof_precedente and len(hof_corrente) > 0:
            contatore_stagnazione += 1
        else:
            # Se la HoF è cambiata, controlliamo se eravamo in una fase di stagnazione lunga
            if contatore_stagnazione >= soglia_stagnazione:
                print(f"⚠️ RISTAGNO RILEVATO!")
                print(f"   Dalla generazione: {inizio_stagnazione}")
                print(f"   Alla generazione:  {gen_precedente}")
                print(f"   Durata: {contatore_stagnazione} generazioni consecutive con gli stessi individui.")
                print("-" * 50)

            # Resetta il contatore per la nuova HoF
            contatore_stagnazione = 1
            inizio_stagnazione = gen_corrente
            hof_precedente = hof_corrente

        gen_precedente = gen_corrente

    # Controllo finale (se l'esecuzione finisce mentre c'è un ristagno in corso)
    if contatore_stagnazione >= soglia_stagnazione:
        print(f"⚠️ RISTAGNO FINALE IN CORSO!")
        print(f"   Dalla generazione {inizio_stagnazione} fino alla fine (gen {gen_precedente}).")
        print(f"   Durata: {contatore_stagnazione} generazioni.")
        print("-" * 50)

    print("\nAnalisi completata.")


# ==========================================
# ESECUZIONE DELLO SCRIPT
# ==========================================
if __name__ == "__main__":
    # Inserisci qui il percorso della cartella dove tieni i JSON
    CARTELLA = "Downloads/Downloads/3500pop_200gen_200res"  # Usa "./" se lo script è nella stessa cartella dei json

    # Quante generazioni consecutive identiche servono per far scattare l'allarme?
    SOGLIA = 5

    analizza_stagnazione_hof(CARTELLA, SOGLIA)