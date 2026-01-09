#!/bin/bash
#SBATCH --job-name=BioIA_Job       # Nome del lavoro
#SBATCH --output=risultato_%j.log  # Nome del file di log (output)
#SBATCH --error=errore_%j.log      # Nome del file per eventuali errori
#SBATCH --partition=edu-long        # O la partizione indicata dai prof
#SBATCH --account=bio.inspired.ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G                   # Memoria RAM richiesta
#SBATCH --time=1-00:00:00           # Limite massimo di tempo (HH:MM:SS)
#SBATCH --gres=gpu:2

# Carica l'ambiente
source ~/.bashrc
conda activate bio_env           # Inserisci qui il nome del tuo ambiente conda

# Esegui il tuo script Python
python3 GP/gp_run.py           # CAMBIA QUESTO con il nome del tuo file .py